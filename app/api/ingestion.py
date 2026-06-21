"""
문서/Git push 인덱싱 엔드포인트.

호출 주체: Spring datasource 도메인. 사용자가 datasource를 등록하면 Spring이
이 엔드포인트를 호출합니다. FastAPI는 202를 즉시 반환하고 BackgroundTasks로
파이프라인을 실행합니다.

POST /ingestion/document — KNOWLEDGE_DOCUMENTS 레코드 생성(analysis_status=PENDING) 후 인덱싱 트리거
POST /ingestion/git      — Git 커밋 인덱싱 트리거
"""

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, Depends, status
from sqlalchemy.orm import Session

from app.core.security import verify_internal_api_key
from app.db.models import KnowledgeDocument
from app.db.session import get_db
from app.pipelines.document_ingestion import ingest_document
from app.pipelines.git_ingestion import ingest_git_commits
from app.schemas.ingestion import DocumentIngestionRequest, GitIngestionRequest

router = APIRouter()


@router.post("/document", status_code=status.HTTP_202_ACCEPTED)
async def ingest_document_endpoint(
    request: DocumentIngestionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: None = Depends(verify_internal_api_key),
) -> dict:
    """문서 인덱싱을 예약합니다. 완료 여부는 KNOWLEDGE_DOCUMENTS.analysis_status로 확인합니다."""
    now = datetime.now(timezone.utc)
    doc = KnowledgeDocument(
        id=str(uuid.uuid4()),
        workspace_id=request.workspace_id,
        source_id=request.source_id,
        title=request.title,
        document_type=request.doc_type,
        file_path=request.file_path,
        analysis_status="PENDING",
        created_at=now,
        updated_at=now,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    background_tasks.add_task(
        ingest_document,
        workspace_id=request.workspace_id,
        knowledge_document_id=doc.id,
        file_path=request.file_path,
        doc_type=request.doc_type,
    )
    return {"knowledge_document_id": doc.id, "status": "PENDING"}


@router.post("/git", status_code=status.HTTP_202_ACCEPTED)
async def ingest_git_endpoint(
    request: GitIngestionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_internal_api_key),
) -> dict:
    """Git 커밋 인덱싱을 예약합니다. 이미 인덱싱된 커밋은 건너뜁니다."""
    background_tasks.add_task(
        ingest_git_commits,
        workspace_id=request.workspace_id,
        source_id=request.source_id,
        commits=request.commits,
    )
    return {"status": "accepted", "commit_count": len(request.commits)}

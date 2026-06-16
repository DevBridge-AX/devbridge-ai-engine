"""
업로드 문서 청킹/임베딩/인덱싱 파이프라인.

흐름: 파일 로드 → chunker → embedder → document_chunks(MySQL) 저장
      + vector_store.add() → KNOWLEDGE_DOCUMENTS.status 갱신 → usage_logs 누적

백그라운드 태스크로 동작합니다. 완료/실패 여부는 KNOWLEDGE_DOCUMENTS.status로 추적합니다.

확인 필요:
- file_path는 FastAPI와 Spring이 공유하는 파일시스템(예: Docker 볼륨)에 위치해야
  합니다. 공유 FS가 없다면 Spring이 파일 내용을 payload로 직접 전달하는 방식으로
  변경이 필요합니다.
- 현재 read_text()로 UTF-8 텍스트만 처리합니다. PDF 등 바이너리 포맷은 별도
  파서 도입이 필요합니다.
- 청킹 중간 vector_store.add() 실패 시 MySQL에는 document_chunks가 남지 않지만
  (rollback), 이미 vector_store에 추가된 항목은 자동으로 제거되지 않습니다.
  완전한 원자성이 필요하면 보상 트랜잭션 패턴 도입이 필요합니다.
"""

import logging
import uuid
from pathlib import Path

from sqlalchemy import select, update

from app.core.embeddings.embedder import embed_texts
from app.core.rag.chunker import chunk_document
from app.db.models import ChunkSourceType, DocumentChunk, KnowledgeDocument
from app.db.session import SessionLocal, log_embedding_usage
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


async def ingest_document(
    workspace_id: int,
    knowledge_document_id: int,
    file_path: str,
    doc_type: str,
) -> None:
    """문서 인덱싱 백그라운드 태스크. 완료 후 status를 indexed/failed로 갱신합니다."""
    with SessionLocal() as db:
        try:
            await _run(db, workspace_id, knowledge_document_id, file_path, doc_type)
            db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == knowledge_document_id)
                .values(status="indexed")
            )
            db.commit()
        except Exception:
            logger.exception(
                "document_ingestion failed: knowledge_document_id=%d", knowledge_document_id
            )
            db.rollback()
            db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == knowledge_document_id)
                .values(status="failed")
            )
            db.commit()


async def _run(
    db,
    workspace_id: int,
    knowledge_document_id: int,
    file_path: str,
    doc_type: str,
) -> None:
    text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    chunks = chunk_document(text, doc_type)
    if not chunks:
        logger.warning("document_ingestion: no chunks produced for id=%d", knowledge_document_id)
        return

    result = await embed_texts([c.content for c in chunks])
    vector_store = get_vector_store()

    pending: list[tuple[DocumentChunk, list[float], str]] = []
    for chunk, embedding in zip(chunks, result.embeddings):
        vector_id = str(uuid.uuid4())
        doc_chunk = DocumentChunk(
            workspace_id=workspace_id,
            source_type=ChunkSourceType.DOCUMENT,
            source_id=knowledge_document_id,
            content=chunk.content,
            chunk_metadata=chunk.chunk_metadata,
            embedding_model=result.embedding_model,
            embedding_model_version=result.embedding_model_version,
            vector_id=vector_id,
        )
        db.add(doc_chunk)
        pending.append((doc_chunk, embedding, vector_id))

    db.flush()

    for doc_chunk, embedding, vector_id in pending:
        vector_store.add(
            vector_id=vector_id,
            chunk_id=doc_chunk.id,
            embedding=embedding,
            metadata={
                "source_type": ChunkSourceType.DOCUMENT.value,
                "source_id": knowledge_document_id,
            },
            workspace_id=workspace_id,
        )

    log_embedding_usage(db, workspace_id, result.embedding_model, result.total_tokens)

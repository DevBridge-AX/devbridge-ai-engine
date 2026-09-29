"""
업로드 문서 청킹/임베딩/인덱싱 파이프라인.

흐름: 파일 로드 → chunker → embedder → document_chunks(MySQL) 저장
      + vector_store.add() → KNOWLEDGE_DOCUMENTS.analysis_status 갱신 → usage_logs 누적

백그라운드 태스크로 동작합니다. 완료/실패 여부는 KNOWLEDGE_DOCUMENTS.analysis_status로 추적합니다.

관측성(R-4): read/chunk/embed/store 구간별 소요 시간과 file_bytes/char_count/chunk_count,
결과(COMPLETED/EMPTY/FAILED)와 실패 단계(failure_stage)/예외 클래스명(error_type)을
app.core.metrics.record_metric("ingestion", ...)으로 기록합니다. 이 메트릭은 별도
집계용이며 KNOWLEDGE_DOCUMENTS.analysis_status 값에는 영향을 주지 않습니다.

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
import time
import uuid
from pathlib import Path

from sqlalchemy import update

from app.core.embeddings.embedder import embed_texts
from app.core.metrics import StageTimer, record_metric
from app.core.rag.bm25_index import get_bm25_manager
from app.core.rag.chunker import chunk_document
from app.db.models import ChunkSourceType, DocumentChunk, KnowledgeDocument
from app.db.session import SessionLocal, log_embedding_usage
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


async def ingest_document(
    workspace_id: str,
    knowledge_document_id: str,
    file_path: str,
    doc_type: str,
    task_id: str | None = None,
    sensitivity_level: str = "normal",
) -> None:
    """문서 인덱싱 백그라운드 태스크. 완료 후 analysis_status를 indexed/failed로 갱신합니다.

    task_id/sensitivity_level은 청크에 접근 제어 스냅샷으로 복사됩니다(docs/access-control.md §3.3).
    """
    with SessionLocal() as db:
        try:
            await _run(
                db,
                workspace_id,
                knowledge_document_id,
                file_path,
                doc_type,
                task_id=task_id,
                sensitivity_level=sensitivity_level,
            )
            db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == knowledge_document_id)
                .values(analysis_status="COMPLETED")
            )
            db.commit()
        except Exception:
            logger.exception(
                "document_ingestion failed: knowledge_document_id=%s", knowledge_document_id
            )
            db.rollback()
            db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == knowledge_document_id)
                .values(analysis_status="FAILED")
            )
            db.commit()


async def _run(
    db,
    workspace_id: str,
    knowledge_document_id: str,
    file_path: str,
    doc_type: str,
    task_id: str | None = None,
    sensitivity_level: str = "normal",
) -> None:
    timer = StageTimer()
    total_start = time.perf_counter()
    file_bytes = _safe_file_size(file_path)

    try:
        with timer.measure("read_ms"):
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=0, chunk_count=0,
            result="FAILED", failure_stage="read", error_type=type(exc).__name__,
        )
        raise

    char_count = len(text)

    try:
        with timer.measure("chunk_ms"):
            chunks = chunk_document(text, doc_type)
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=0,
            result="FAILED", failure_stage="chunk", error_type=type(exc).__name__,
        )
        raise

    if not chunks:
        logger.warning("document_ingestion: no chunks produced for id=%s", knowledge_document_id)
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=0,
            result="EMPTY", failure_stage=None, error_type=None,
        )
        return

    try:
        with timer.measure("embed_ms"):
            result = await embed_texts([c.content for c in chunks])
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=len(chunks),
            result="FAILED", failure_stage="embed", error_type=type(exc).__name__,
        )
        raise

    vector_store = get_vector_store()

    try:
        with timer.measure("store_ms"):
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
                    task_id=task_id,
                    sensitivity_level=sensitivity_level,
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

            get_bm25_manager().invalidate(workspace_id)
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=len(chunks),
            result="FAILED", failure_stage="store", error_type=type(exc).__name__,
        )
        raise

    _record_ingestion_metric(
        timer, total_start, workspace_id, knowledge_document_id, doc_type,
        file_bytes=file_bytes, char_count=char_count, chunk_count=len(chunks),
        result="COMPLETED", failure_stage=None, error_type=None,
    )


# ---------------------------------------------------------------------------
# 관측성 헬퍼
# ---------------------------------------------------------------------------

def _safe_file_size(file_path: str) -> int | None:
    """파일 크기(byte)를 반환합니다. 조회 실패 시 None(메트릭 흐름을 막지 않음)."""
    try:
        return Path(file_path).stat().st_size
    except OSError:
        return None


def _record_ingestion_metric(
    timer: StageTimer,
    total_start: float,
    workspace_id: str,
    knowledge_document_id: str,
    doc_type: str,
    *,
    file_bytes: int | None,
    char_count: int,
    chunk_count: int,
    result: str,
    failure_stage: str | None,
    error_type: str | None,
) -> None:
    """문서 인덱싱 1건의 지표를 기록합니다. 문서 원문은 포함하지 않습니다."""
    total_ms = (time.perf_counter() - total_start) * 1000
    payload = {
        "workspace_id": workspace_id,
        "knowledge_document_id": knowledge_document_id,
        "doc_type": doc_type,
        "file_bytes": file_bytes,
        "char_count": char_count,
        "chunk_count": chunk_count,
        "read_ms": timer.stages.get("read_ms"),
        "chunk_ms": timer.stages.get("chunk_ms"),
        "embed_ms": timer.stages.get("embed_ms"),
        "store_ms": timer.stages.get("store_ms"),
        "total_ms": total_ms,
        "result": result,
        "failure_stage": failure_stage,
        "error_type": error_type,
    }
    record_metric("ingestion", payload)

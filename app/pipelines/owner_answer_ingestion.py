"""
담당자 답변(Q&A) 청킹/임베딩/인덱싱 파이프라인.

흐름: "질문: {question}\n\n답변: {answer}" 합성 → chunker → embedder
      → document_chunks(MySQL) 저장 + vector_store.add() → usage_logs 누적

백그라운드 태스크로 동작합니다.
OWNER_CONFIRMATIONS 테이블은 Spring 소유이므로 직접 조회하지 않으며,
필요한 데이터는 모두 API payload로 전달받습니다.
"""

import logging
import uuid

from app.core.embeddings.embedder import embed_texts
from app.core.rag.bm25_index import get_bm25_manager
from app.core.rag.chunker import chunk_document
from app.db.models import ChunkSourceType, DocumentChunk
from app.db.session import SessionLocal, log_embedding_usage
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)


async def ingest_owner_answer(
    workspace_id: str,
    confirmation_id: str,
    question: str,
    answer: str,
    owner_employee_id: str,
    owner_name: str,
) -> None:
    """담당자 답변 인덱싱 백그라운드 태스크."""
    with SessionLocal() as db:
        try:
            await _run(
                db=db,
                workspace_id=workspace_id,
                confirmation_id=confirmation_id,
                question=question,
                answer=answer,
                owner_employee_id=owner_employee_id,
                owner_name=owner_name,
            )
            db.commit()
        except Exception:
            logger.exception(
                "owner_answer_ingestion failed: confirmation_id=%s",
                confirmation_id,
            )
            db.rollback()


async def _run(
    db,
    workspace_id: str,
    confirmation_id: str,
    question: str,
    answer: str,
    owner_employee_id: str,
    owner_name: str,
) -> None:
    text = f"질문: {question}\n\n답변: {answer}"
    chunks = chunk_document(text, doc_type="text")
    if not chunks:
        logger.warning(
            "owner_answer_ingestion: no chunks produced for confirmation_id=%s",
            confirmation_id,
        )
        return

    result = await embed_texts([c.content for c in chunks])
    vector_store = get_vector_store()

    pending: list[tuple[DocumentChunk, list[float], str]] = []
    for chunk, embedding in zip(chunks, result.embeddings):
        vector_id = str(uuid.uuid4())
        doc_chunk = DocumentChunk(
            workspace_id=workspace_id,
            source_type=ChunkSourceType.OWNER_ANSWER,
            source_id=confirmation_id,
            content=chunk.content,
            chunk_metadata={
                **(chunk.chunk_metadata or {}),
                "owner_employee_id": owner_employee_id,
                "owner_name": owner_name,
                "question": question,
            },
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
                "source_type": ChunkSourceType.OWNER_ANSWER.value,
                "source_id": confirmation_id,
            },
            workspace_id=workspace_id,
        )

    log_embedding_usage(db, workspace_id, result.embedding_model, result.total_tokens)

    get_bm25_manager().invalidate(workspace_id)

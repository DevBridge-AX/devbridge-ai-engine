"""
document_chunks 벡터 검색.

흐름: embed_texts([query]) → vector_store.search() → chunk_id로 document_chunks 조회
      → source_type별 원본 레코드(KNOWLEDGE_DOCUMENTS / GIT_COMMITS / DATABASE_SCHEMAS)
      에서 title·author_id 추출 → RetrievedChunk 반환.

유사도 점수는 grounding.py의 1차 필터 및 suggested_owner_id 결정에 사용됩니다.
"""

from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.embeddings.embedder import embed_texts
from app.db.models import (
    ChunkSourceType,
    DatabaseSchema,
    DocumentChunk,
    GitCommit,
    KnowledgeDocument,
)
from app.db.vector_store import get_vector_store


@dataclass
class RetrievedChunk:
    chunk_id: int
    source_type: str
    source_id: int
    title: str
    content: str
    similarity_score: float
    author_id: str | None = field(default=None)  # GIT_COMMIT 청크에만 설정됨


async def retrieve(
    query: str,
    workspace_id: str,
    db: Session,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """쿼리에 대한 관련 document_chunks를 유사도 순으로 반환합니다."""
    embed_result = await embed_texts([query])
    query_embedding = embed_result.embeddings[0]

    raw_results = get_vector_store().search(query_embedding, workspace_id=workspace_id, top_k=top_k)
    if not raw_results:
        return []

    chunk_ids = [r["chunk_id"] for r in raw_results]
    score_by_id = {r["chunk_id"]: r["score"] for r in raw_results}

    chunks = db.execute(
        select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
    ).scalars().all()
    chunk_by_id = {c.id: c for c in chunks}

    results: list[RetrievedChunk] = []
    for chunk_id in chunk_ids:
        chunk = chunk_by_id.get(chunk_id)
        if chunk is None:
            continue
        title, author_id = _resolve_title_and_author(chunk, db)
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                source_type=chunk.source_type.value,
                source_id=chunk.source_id,
                title=title,
                content=chunk.content,
                similarity_score=score_by_id[chunk_id],
                author_id=author_id,
            )
        )

    return results


def _resolve_title_and_author(
    chunk: DocumentChunk, db: Session
) -> tuple[str, str | None]:
    """source_type에 따라 표시용 title과 author_id(git_commit 전용)를 반환합니다."""
    source_type = chunk.source_type

    if source_type == ChunkSourceType.DOCUMENT:
        row = db.execute(
            select(KnowledgeDocument).where(KnowledgeDocument.id == chunk.source_id)
        ).scalar_one_or_none()
        title = row.title if row else f"Document #{chunk.source_id}"
        return title, None

    if source_type == ChunkSourceType.GIT_COMMIT:
        row = db.execute(
            select(GitCommit).where(GitCommit.id == chunk.source_id)
        ).scalar_one_or_none()
        if row:
            title = row.commit_message.split("\n")[0][:72]
            return title, row.author_id
        return f"Commit #{chunk.source_id}", None

    if source_type == ChunkSourceType.DB_SCHEMA:
        row = db.execute(
            select(DatabaseSchema).where(DatabaseSchema.id == chunk.source_id)
        ).scalar_one_or_none()
        title = f"{row.schema_name}.{row.table_name}" if row else f"Schema #{chunk.source_id}"
        return title, None

    if source_type == ChunkSourceType.OWNER_ANSWER:
        meta = chunk.chunk_metadata or {}
        owner_name = meta.get("owner_name", "담당자")
        return f"담당자 답변: {owner_name}", None

    return f"Chunk #{chunk.source_id}", None

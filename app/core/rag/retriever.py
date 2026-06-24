"""
document_chunks 하이브리드 검색 (벡터 + BM25).

bm25_enabled=True (기본값):
  embed_texts → asyncio.create_task(embedding)
  → BM25 검색(동기, 캐시 히트 시 <1ms)
  → await embedding → Chroma 벡터 검색
  → RRF 병합 → top_k 반환

bm25_enabled=False:
  기존 vector-only 경로 그대로 동작.

similarity_score는 원본 cosine 유사도를 유지합니다 (grounding.py 호환).
BM25-only 히트는 similarity_score=0.0으로, grounding threshold에서 자연 필터링됩니다.
"""

import asyncio
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.embeddings.embedder import embed_texts
from app.core.rag.bm25_index import BM25SearchResult, get_bm25_manager
from app.db.models import (
    ChunkSourceType,
    DatabaseSchema,
    DocumentChunk,
    GitCommit,
    KnowledgeDocument,
)
from app.db.vector_store import get_vector_store

_OVERFETCH_FACTOR = 3
_RRF_K = 60


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
    """쿼리에 대한 관련 document_chunks를 유사도 순으로 반환합니다.

    bm25_enabled=True일 때 하이브리드 검색(BM25 + Vector, RRF 병합)을 수행합니다.
    """
    settings = get_settings()

    if not settings.bm25_enabled:
        return await _vector_only_retrieve(query, workspace_id, db, top_k)

    overfetch_k = top_k * _OVERFETCH_FACTOR

    embed_task = asyncio.create_task(_get_query_embedding(query))
    bm25_results = _search_bm25_sync(query, workspace_id, db, overfetch_k)
    query_embedding = await embed_task

    vector_results = get_vector_store().search(
        query_embedding, workspace_id=workspace_id, top_k=overfetch_k
    )

    merged = _rrf_merge(vector_results, bm25_results, top_k)
    if not merged:
        return []

    return _build_retrieved_chunks(merged, db)


async def _vector_only_retrieve(
    query: str,
    workspace_id: str,
    db: Session,
    top_k: int,
) -> list[RetrievedChunk]:
    """기존 vector-only 검색 경로."""
    embed_result = await embed_texts([query])
    query_embedding = embed_result.embeddings[0]

    raw_results = get_vector_store().search(
        query_embedding, workspace_id=workspace_id, top_k=top_k
    )
    if not raw_results:
        return []

    chunk_ids = [r["chunk_id"] for r in raw_results]
    score_by_id = {r["chunk_id"]: r["score"] for r in raw_results}

    chunks = (
        db.execute(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))
        .scalars()
        .all()
    )
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


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


async def _get_query_embedding(query: str) -> list[float]:
    embed_result = await embed_texts([query])
    return embed_result.embeddings[0]


def _search_bm25_sync(
    query: str,
    workspace_id: str,
    db: Session,
    top_k: int,
) -> list[BM25SearchResult]:
    return get_bm25_manager().search(query, workspace_id, db, top_k)


def _rrf_merge(
    vector_results: list[dict],
    bm25_results: list[BM25SearchResult],
    top_k: int,
) -> list[dict]:
    """RRF (Reciprocal Rank Fusion) 병합.

    반환: [{"chunk_id": int, "rrf_score": float, "vector_score": float}]
    vector_score는 원본 cosine 유사도. BM25-only 히트는 vector_score=0.0.
    """
    rrf_scores: dict[int, float] = {}
    vector_scores: dict[int, float] = {}

    for rank, item in enumerate(vector_results):
        cid = item["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
        vector_scores[cid] = item["score"]

    for item in bm25_results:
        cid = item.chunk_id
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (
            _RRF_K + item.bm25_rank + 1
        )
        vector_scores.setdefault(cid, 0.0)

    sorted_ids = sorted(
        rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True
    )[:top_k]

    return [
        {
            "chunk_id": cid,
            "rrf_score": rrf_scores[cid],
            "vector_score": vector_scores[cid],
        }
        for cid in sorted_ids
    ]


def _build_retrieved_chunks(
    merged: list[dict], db: Session
) -> list[RetrievedChunk]:
    """RRF 병합 결과를 RetrievedChunk 리스트로 변환합니다."""
    chunk_ids = [m["chunk_id"] for m in merged]
    score_by_id = {m["chunk_id"]: m["vector_score"] for m in merged}

    chunks = (
        db.execute(select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids)))
        .scalars()
        .all()
    )
    chunk_by_id = {c.id: c for c in chunks}

    results: list[RetrievedChunk] = []
    for item in merged:
        chunk = chunk_by_id.get(item["chunk_id"])
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
                similarity_score=score_by_id[chunk.id],
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
        title = (
            f"{row.schema_name}.{row.table_name}"
            if row
            else f"Schema #{chunk.source_id}"
        )
        return title, None

    if source_type == ChunkSourceType.OWNER_ANSWER:
        meta = chunk.chunk_metadata or {}
        owner_name = meta.get("owner_name", "담당자")
        return f"담당자 답변: {owner_name}", None

    return f"Chunk #{chunk.source_id}", None

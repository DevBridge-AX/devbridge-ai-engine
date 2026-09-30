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

접근 제어(docs/access-control.md §6): access가 주어지면 overfetch된 벡터·BM25 후보를
RRF 병합 전에 document_chunks의 task_id/sensitivity_level 스냅샷으로 사후 필터링합니다.
병합 후에 거르면 최종 결과가 top_k보다 줄어들기 때문입니다.

관측성: timer(app.core.metrics.StageTimer)가 주어지면 embed_ms/bm25_ms/vector_ms/
acl_filter_ms 구간과 acl_refetch_count(0/1) 필드를 기록합니다. 미전달 시(None) 기존 동작과 동일합니다.

ACL 재조회: 필터 후 결과가 top_k 미만이고 필터 전 후보 페이지가 가득 찼다면(더 있을 수 있음)
overfetch 배수를 2배(상한 12)로 늘려 1회만 재검색합니다. 쿼리 임베딩은 재사용합니다.
"""

import asyncio
from contextlib import nullcontext
from dataclasses import dataclass, field

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.embeddings.embedder import embed_texts
from app.core.metrics import StageTimer
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
_OVERFETCH_MAX_FACTOR = 12
_ACL_MAX_REFETCH = 1
_RRF_K = 60


@dataclass(frozen=True)
class AccessFilter:
    """검색 단계에서 강제할 접근 범위. 권한 계산은 Spring 책임이며 이 레포는 강제만 합니다.

    - accessible_task_ids: None이면 task 제한 없음. 리스트면 task 미지정 청크 + 해당 task 청크만 허용.
    - can_view_restricted: False면 sensitivity_level='restricted' 청크 제외.
    """

    accessible_task_ids: list[str] | None = None
    can_view_restricted: bool = False

    @property
    def is_unrestricted(self) -> bool:
        return self.accessible_task_ids is None and self.can_view_restricted


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
    access: AccessFilter | None = None,
    timer: StageTimer | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievedChunk]:
    """쿼리에 대한 관련 document_chunks를 유사도 순으로 반환합니다.

    bm25_enabled=True일 때 하이브리드 검색(BM25 + Vector, RRF 병합)을 수행합니다.
    access가 None이면 접근 제어 없이 검색합니다.
    timer가 주어지면 embed_ms/bm25_ms/vector_ms/acl_filter_ms 구간을 기록합니다.
    query_embedding이 주어지면 쿼리 임베딩 호출을 생략하고 그 값을 재사용합니다
    (시맨틱 캐시 조회에서 이미 계산한 임베딩).
    """
    settings = get_settings()

    if not settings.bm25_enabled:
        return await _vector_only_retrieve(
            query, workspace_id, db, top_k, access, timer, query_embedding
        )

    factor = _OVERFETCH_FACTOR
    overfetch_k = top_k * factor
    restricted = access is not None and not access.is_unrestricted

    embed_task = (
        asyncio.create_task(_get_query_embedding(query, timer)) if query_embedding is None else None
    )
    bm25_raw = _search_bm25_sync(query, workspace_id, db, overfetch_k, timer)
    if embed_task is not None:
        query_embedding = await embed_task

    refetch_count = 0
    while True:
        with timer.measure("vector_ms") if timer else nullcontext():
            vector_raw = get_vector_store().search(
                query_embedding, workspace_id=workspace_id, top_k=overfetch_k
            )

        vector_results, bm25_results = vector_raw, bm25_raw
        if restricted:
            with timer.measure("acl_filter_ms") if timer else nullcontext():
                allowed = _allowed_chunk_ids(
                    {r["chunk_id"] for r in vector_raw} | {r.chunk_id for r in bm25_raw},
                    access,
                    db,
                )
                vector_results = [r for r in vector_raw if r["chunk_id"] in allowed]
                bm25_results = [r for r in bm25_raw if r.chunk_id in allowed]

        merged = _rrf_merge(vector_results, bm25_results, top_k)

        candidates_full = len(vector_raw) >= overfetch_k or len(bm25_raw) >= overfetch_k
        next_factor = min(factor * 2, _OVERFETCH_MAX_FACTOR)
        if not (
            restricted
            and len(merged) < top_k
            and candidates_full
            and refetch_count < _ACL_MAX_REFETCH
            and next_factor > factor
        ):
            break
        refetch_count += 1
        factor = next_factor
        overfetch_k = top_k * factor
        bm25_raw = _search_bm25_sync(query, workspace_id, db, overfetch_k, timer)

    if timer:
        timer.set_field("acl_refetch_count", refetch_count)

    if not merged:
        return []

    return _build_retrieved_chunks(merged, db)


async def _vector_only_retrieve(
    query: str,
    workspace_id: str,
    db: Session,
    top_k: int,
    access: AccessFilter | None = None,
    timer: StageTimer | None = None,
    query_embedding: list[float] | None = None,
) -> list[RetrievedChunk]:
    """기존 vector-only 검색 경로."""
    if query_embedding is None:
        with timer.measure("embed_ms") if timer else nullcontext():
            embed_result = await embed_texts([query])
        query_embedding = embed_result.embeddings[0]

    restricted = access is not None and not access.is_unrestricted
    factor = _OVERFETCH_FACTOR
    refetch_count = 0
    while True:
        fetch_k = top_k * factor if restricted else top_k
        with timer.measure("vector_ms") if timer else nullcontext():
            vector_raw = get_vector_store().search(
                query_embedding, workspace_id=workspace_id, top_k=fetch_k
            )
        raw_results = vector_raw
        if restricted:
            with timer.measure("acl_filter_ms") if timer else nullcontext():
                allowed = _allowed_chunk_ids({r["chunk_id"] for r in vector_raw}, access, db)
                raw_results = [r for r in vector_raw if r["chunk_id"] in allowed][:top_k]

        next_factor = min(factor * 2, _OVERFETCH_MAX_FACTOR)
        if not (
            restricted
            and len(raw_results) < top_k
            and len(vector_raw) >= fetch_k
            and refetch_count < _ACL_MAX_REFETCH
            and next_factor > factor
        ):
            break
        refetch_count += 1
        factor = next_factor

    if timer:
        timer.set_field("acl_refetch_count", refetch_count)
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


async def _get_query_embedding(query: str, timer: StageTimer | None = None) -> list[float]:
    with timer.measure("embed_ms") if timer else nullcontext():
        embed_result = await embed_texts([query])
    return embed_result.embeddings[0]


def _search_bm25_sync(
    query: str,
    workspace_id: str,
    db: Session,
    top_k: int,
    timer: StageTimer | None = None,
) -> list[BM25SearchResult]:
    with timer.measure("bm25_ms") if timer else nullcontext():
        return get_bm25_manager().search(query, workspace_id, db, top_k)


def _allowed_chunk_ids(
    candidate_ids: set[int],
    access: AccessFilter,
    db: Session,
) -> set[int]:
    """후보 chunk_id 중 access 범위에 드는 id만 한 번의 쿼리로 추려 반환합니다."""
    if not candidate_ids:
        return set()

    stmt = select(DocumentChunk.id).where(DocumentChunk.id.in_(candidate_ids))
    if access.accessible_task_ids is not None:
        stmt = stmt.where(
            or_(
                DocumentChunk.task_id.is_(None),
                DocumentChunk.task_id.in_(access.accessible_task_ids),
            )
        )
    if not access.can_view_restricted:
        stmt = stmt.where(DocumentChunk.sensitivity_level != "restricted")

    return set(db.execute(stmt).scalars().all())


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

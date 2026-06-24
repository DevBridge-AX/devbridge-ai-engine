"""
워크스페이스별 BM25 인덱스 관리.

rank_bm25.BM25Okapi를 workspace_id별로 관리하며, TTL 기반 캐시 + 명시적
무효화를 지원합니다. document_chunks 테이블에서 (id, content) 쌍을 로드하여
BM25 인덱스를 구축합니다.
"""

import logging
import threading
import time
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.rag.tokenizer import tokenize_for_bm25
from app.db.models import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class _CachedIndex:
    bm25: BM25Okapi
    chunk_ids: list[int]
    created_at: float
    chunk_count: int = field(default=0)


@dataclass
class BM25SearchResult:
    chunk_id: int
    bm25_rank: int


class BM25IndexManager:
    """워크스페이스별 BM25 인덱스 싱글톤 매니저."""

    def __init__(self) -> None:
        self._indices: dict[str, _CachedIndex] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, workspace_id: str) -> threading.Lock:
        with self._global_lock:
            if workspace_id not in self._locks:
                self._locks[workspace_id] = threading.Lock()
            return self._locks[workspace_id]

    def search(
        self,
        query: str,
        workspace_id: str,
        db: Session,
        top_k: int = 15,
    ) -> list[BM25SearchResult]:
        """BM25 검색. 캐시 미스 시 DB에서 로드하여 인덱스를 구축합니다."""
        lock = self._get_lock(workspace_id)
        with lock:
            index = self._get_or_build(workspace_id, db)

        if index is None or index.chunk_count == 0:
            return []

        tokens = tokenize_for_bm25(query)
        if not tokens:
            return []

        scores = index.bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            BM25SearchResult(chunk_id=index.chunk_ids[i], bm25_rank=rank)
            for rank, i in enumerate(top_indices)
            if scores[i] > 0
        ]

    def invalidate(self, workspace_id: str) -> None:
        """워크스페이스 BM25 인덱스를 무효화합니다."""
        lock = self._get_lock(workspace_id)
        with lock:
            self._indices.pop(workspace_id, None)
        logger.info("BM25 index invalidated: workspace_id=%s", workspace_id)

    def invalidate_all(self) -> None:
        """모든 워크스페이스 BM25 인덱스를 무효화합니다."""
        with self._global_lock:
            self._indices.clear()
        logger.info("All BM25 indices invalidated")

    def _get_or_build(
        self, workspace_id: str, db: Session
    ) -> _CachedIndex | None:
        settings = get_settings()
        ttl = settings.bm25_cache_ttl_seconds

        cached = self._indices.get(workspace_id)
        if cached is not None:
            age = time.monotonic() - cached.created_at
            if ttl <= 0 or age < ttl:
                return cached

        return self._build_index(workspace_id, db)

    def _build_index(
        self, workspace_id: str, db: Session
    ) -> _CachedIndex | None:
        rows = db.execute(
            select(DocumentChunk.id, DocumentChunk.content).where(
                DocumentChunk.workspace_id == workspace_id
            )
        ).all()

        if not rows:
            return None

        chunk_ids: list[int] = []
        corpus: list[list[str]] = []
        for row in rows:
            tokens = tokenize_for_bm25(row[1])
            if tokens:
                chunk_ids.append(row[0])
                corpus.append(tokens)

        if not corpus:
            return None

        bm25 = BM25Okapi(corpus)

        index = _CachedIndex(
            bm25=bm25,
            chunk_ids=chunk_ids,
            created_at=time.monotonic(),
            chunk_count=len(chunk_ids),
        )
        self._indices[workspace_id] = index

        logger.info(
            "BM25 index built: workspace_id=%s, chunks=%d",
            workspace_id,
            len(chunk_ids),
        )
        return index


_manager: BM25IndexManager | None = None
_init_lock = threading.Lock()


def get_bm25_manager() -> BM25IndexManager:
    """BM25IndexManager 싱글톤을 반환합니다."""
    global _manager
    if _manager is None:
        with _init_lock:
            if _manager is None:
                _manager = BM25IndexManager()
    return _manager

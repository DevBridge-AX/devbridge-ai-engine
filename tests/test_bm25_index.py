"""
app/core/rag/bm25_index.py 유닛 테스트.

DB 의존성을 모킹하여 BM25IndexManager의 빌드/검색/캐시/무효화를 검증합니다.
"""

import time
from collections import namedtuple
from unittest.mock import MagicMock, patch

import pytest

from app.config import get_settings
from app.core.rag.bm25_index import BM25IndexManager

MockRow = namedtuple("MockRow", ["id", "content"])

SAMPLE_CHUNKS_WS1 = [
    MockRow(1, "FastAPI 인증 흐름과 JWT 토큰 검증 방법"),
    MockRow(2, "데이터베이스 스키마 마이그레이션 가이드"),
    MockRow(3, "Git 커밋 컨벤션 및 브랜치 전략"),
    MockRow(4, "Spring Boot REST API 엔드포인트 설계"),
    MockRow(5, "RAG 파이프라인 벡터 검색 최적화"),
]

SAMPLE_CHUNKS_WS2 = [
    MockRow(10, "프론트엔드 Vue 컴포넌트 구조와 상태 관리"),
    MockRow(11, "CSS 디자인 시스템 토큰 정의"),
    MockRow(12, "백엔드 API 엔드포인트 설계 가이드"),
    MockRow(13, "데이터베이스 인덱스 최적화 방법론"),
]


def _make_mock_db(rows: list[MockRow]) -> MagicMock:
    db = MagicMock()
    mock_result = MagicMock()
    mock_result.all.return_value = [(r.id, r.content) for r in rows]
    db.execute.return_value = mock_result
    return db


class TestBM25IndexManager:

    def test_build_and_search(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        results = manager.search("인증 JWT 토큰", "ws1", db, top_k=3)

        assert len(results) >= 1
        chunk_ids = [r.chunk_id for r in results]
        assert 1 in chunk_ids

    def test_search_returns_ranked(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        results = manager.search("데이터베이스 스키마", "ws1", db, top_k=5)

        assert len(results) >= 1
        assert results[0].bm25_rank == 0
        for i, r in enumerate(results):
            assert r.bm25_rank == i

    def test_workspace_isolation(self):
        manager = BM25IndexManager()
        db_ws1 = _make_mock_db(SAMPLE_CHUNKS_WS1)
        db_ws2 = _make_mock_db(SAMPLE_CHUNKS_WS2)

        results_ws1 = manager.search("프론트엔드 컴포넌트", "ws1", db_ws1, top_k=5)
        results_ws2 = manager.search("프론트엔드 컴포넌트", "ws2", db_ws2, top_k=5)

        ws1_ids = {r.chunk_id for r in results_ws1}
        ws2_ids = {r.chunk_id for r in results_ws2}
        # ws1에는 id 10, 11이 없어야 함
        assert ws1_ids.isdisjoint({10, 11})
        # ws2에서는 프론트엔드 관련 청크가 반환되어야 함
        assert len(results_ws2) >= 1
        assert all(r.chunk_id in {10, 11} for r in results_ws2)

    def test_cache_hit(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        manager.search("인증", "ws1", db, top_k=3)
        db.execute.reset_mock()

        manager.search("인증", "ws1", db, top_k=3)
        db.execute.assert_not_called()

    def test_ttl_expiration(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        manager.search("인증", "ws1", db, top_k=3)
        # Force TTL expiration by backdating created_at
        cached = manager._indices.get("ws1")
        assert cached is not None
        cached.created_at = time.monotonic() - 9999

        db.execute.reset_mock()
        manager.search("인증", "ws1", db, top_k=3)
        db.execute.assert_called_once()

    def test_invalidate(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        manager.search("인증", "ws1", db, top_k=3)
        db.execute.reset_mock()

        manager.invalidate("ws1")

        manager.search("인증", "ws1", db, top_k=3)
        db.execute.assert_called_once()

    def test_invalidate_all(self):
        manager = BM25IndexManager()
        db_ws1 = _make_mock_db(SAMPLE_CHUNKS_WS1)
        db_ws2 = _make_mock_db(SAMPLE_CHUNKS_WS2)

        manager.search("인증", "ws1", db_ws1, top_k=3)
        manager.search("Vue", "ws2", db_ws2, top_k=3)
        db_ws1.execute.reset_mock()
        db_ws2.execute.reset_mock()

        manager.invalidate_all()

        manager.search("인증", "ws1", db_ws1, top_k=3)
        manager.search("Vue", "ws2", db_ws2, top_k=3)
        db_ws1.execute.assert_called_once()
        db_ws2.execute.assert_called_once()

    def test_empty_workspace(self):
        manager = BM25IndexManager()
        db = _make_mock_db([])

        results = manager.search("아무거나", "empty_ws", db, top_k=5)
        assert results == []

    def test_empty_query(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        results = manager.search("", "ws1", db, top_k=5)
        assert results == []

    def test_no_term_overlap(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        results = manager.search("zzzzxxx completely unrelated", "ws1", db, top_k=5)
        assert results == []

    def test_top_k_respected(self):
        manager = BM25IndexManager()
        db = _make_mock_db(SAMPLE_CHUNKS_WS1)

        results = manager.search("API", "ws1", db, top_k=2)
        assert len(results) <= 2

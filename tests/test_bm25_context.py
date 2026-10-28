"""
BM25 색인의 청크 문맥 포함 옵션(chunk_context_bm25) 테스트.

플래그 off면 content만, on이면 chunk_metadata.context를 앞에 붙여 색인하는지 검증합니다.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from app.core.rag import bm25_index
from app.core.rag.bm25_index import BM25IndexManager

ROWS = [
    (1, "서버 재시작 절차를 설명한다", {"context": "스테이징 환경 배포 가이드의 롤백 절"}),
    (2, "데이터베이스 백업 주기와 보관 정책", {"context": None}),
    (3, "로그 수집 설정 방법", None),
]


def _db() -> MagicMock:
    db = MagicMock()
    result = MagicMock()
    result.all.return_value = ROWS
    db.execute.return_value = result
    return db


def _patch_flag(monkeypatch, enabled: bool):
    monkeypatch.setattr(
        bm25_index,
        "get_settings",
        lambda: SimpleNamespace(chunk_context_bm25=enabled, bm25_cache_ttl_seconds=600),
    )


def _plain_rows_db() -> MagicMock:
    """flag off일 때 쿼리는 (id, content) 2열만 읽으므로 2-튜플 행을 돌려준다."""
    db = MagicMock()
    result = MagicMock()
    result.all.return_value = [(r[0], r[1]) for r in ROWS]
    db.execute.return_value = result
    return db


def test_context_excluded_by_default(monkeypatch):
    _patch_flag(monkeypatch, False)
    manager = BM25IndexManager()

    results = manager.search("스테이징 롤백", "ws", _plain_rows_db(), top_k=3)

    assert results == []


def test_context_included_when_flag_on(monkeypatch):
    _patch_flag(monkeypatch, True)
    manager = BM25IndexManager()

    results = manager.search("스테이징 롤백", "ws", _db(), top_k=3)

    assert [r.chunk_id for r in results][:1] == [1]


def test_flag_on_without_context_keeps_content_only(monkeypatch):
    _patch_flag(monkeypatch, True)
    manager = BM25IndexManager()

    results = manager.search("로그 수집", "ws", _db(), top_k=3)

    assert [r.chunk_id for r in results][:1] == [3]


def test_flag_on_reads_chunk_metadata_column(monkeypatch):
    _patch_flag(monkeypatch, True)
    db = _db()

    BM25IndexManager().search("백업", "ws", db, top_k=1)

    stmt = str(db.execute.call_args[0][0])
    assert "chunk_metadata" in stmt


def test_flag_off_does_not_read_chunk_metadata_column(monkeypatch):
    _patch_flag(monkeypatch, False)
    db = _plain_rows_db()

    BM25IndexManager().search("백업", "ws", db, top_k=1)

    stmt = str(db.execute.call_args[0][0])
    assert "chunk_metadata" not in stmt

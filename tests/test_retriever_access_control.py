"""
retriever 접근 제어 필터 유닛 테스트 (docs/access-control.md §6).

task_id/sensitivity_level 스냅샷 기준으로 후보가 걸러지는지, 필터가 RRF 병합 전에
적용되어 허용된 청크로 top_k가 채워지는지 검증합니다.
임베딩 API·Chroma·BM25는 mock, DB는 인메모리 SQLite를 사용합니다.
"""

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.rag import retriever
from app.core.rag.bm25_index import BM25SearchResult
from app.core.rag.retriever import AccessFilter, _allowed_chunk_ids
from app.db.models import ChunkSourceType, DocumentChunk

# id: (task_id, sensitivity_level)
_CHUNKS = {
    1: (None, "normal"),        # 워크스페이스 공용
    2: ("task-a", "normal"),
    3: ("task-b", "normal"),
    4: (None, "restricted"),    # 민감 데이터소스
    5: ("task-a", "restricted"),
}


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    DocumentChunk.__table__.create(engine)
    with Session(engine) as session:
        for cid, (task_id, level) in _CHUNKS.items():
            session.add(
                DocumentChunk(
                    id=cid,
                    workspace_id="ws",
                    source_type=ChunkSourceType.DOCUMENT,
                    source_id=f"doc-{cid}",
                    content=f"content {cid}",
                    embedding_model="fake",
                    embedding_model_version="v0",
                    vector_id=f"v{cid}",
                    task_id=task_id,
                    sensitivity_level=level,
                )
            )
        session.commit()
        yield session


class TestAllowedChunkIds:

    def test_task_scope_keeps_unassigned_and_accessible(self, db):
        access = AccessFilter(accessible_task_ids=["task-a"], can_view_restricted=True)
        assert _allowed_chunk_ids(set(_CHUNKS), access, db) == {1, 2, 4, 5}

    def test_empty_task_list_keeps_only_unassigned(self, db):
        access = AccessFilter(accessible_task_ids=[], can_view_restricted=True)
        assert _allowed_chunk_ids(set(_CHUNKS), access, db) == {1, 4}

    def test_restricted_excluded_without_permission(self, db):
        access = AccessFilter(accessible_task_ids=None, can_view_restricted=False)
        assert _allowed_chunk_ids(set(_CHUNKS), access, db) == {1, 2, 3}

    def test_both_axes_combined(self, db):
        access = AccessFilter(accessible_task_ids=["task-a"], can_view_restricted=False)
        assert _allowed_chunk_ids(set(_CHUNKS), access, db) == {1, 2}

    def test_empty_candidates(self, db):
        assert _allowed_chunk_ids(set(), AccessFilter(), db) == set()

    def test_unrestricted_flag(self):
        assert AccessFilter(accessible_task_ids=None, can_view_restricted=True).is_unrestricted
        assert not AccessFilter().is_unrestricted


class TestRetrieveWithAccess:

    @pytest.fixture(autouse=True)
    def patch_deps(self, monkeypatch):
        async def fake_embed(texts):
            return SimpleNamespace(embeddings=[[0.0]])

        vector_hits = [{"chunk_id": cid, "score": 0.9 - cid * 0.1} for cid in (5, 4, 3, 2, 1)]
        bm25_hits = [BM25SearchResult(chunk_id=cid, bm25_rank=i) for i, cid in enumerate((4, 5, 3))]

        monkeypatch.setattr(retriever, "embed_texts", fake_embed)
        monkeypatch.setattr(
            retriever, "get_settings", lambda: SimpleNamespace(bm25_enabled=True)
        )
        monkeypatch.setattr(
            retriever,
            "get_vector_store",
            lambda: SimpleNamespace(search=lambda emb, workspace_id, top_k: vector_hits[:top_k]),
        )
        monkeypatch.setattr(
            retriever,
            "get_bm25_manager",
            lambda: SimpleNamespace(search=lambda q, ws, db, k: bm25_hits[:k]),
        )
        monkeypatch.setattr(
            retriever, "_resolve_title_and_author", lambda chunk, db: (f"t{chunk.id}", None)
        )

    def test_no_access_returns_everything(self, db):
        result = asyncio.run(retriever.retrieve("q", "ws", db, top_k=5))
        assert {c.chunk_id for c in result} == set(_CHUNKS)

    def test_filtered_before_merge_fills_top_k(self, db):
        """restricted·타 task 청크가 상위에 몰려 있어도 허용 청크로 top_k를 채운다."""
        access = AccessFilter(accessible_task_ids=["task-a"], can_view_restricted=False)
        result = asyncio.run(retriever.retrieve("q", "ws", db, top_k=2, access=access))
        assert {c.chunk_id for c in result} == {1, 2}

    def test_default_access_hides_restricted_only(self, db):
        result = asyncio.run(retriever.retrieve("q", "ws", db, top_k=5, access=AccessFilter()))
        assert {c.chunk_id for c in result} == {1, 2, 3}

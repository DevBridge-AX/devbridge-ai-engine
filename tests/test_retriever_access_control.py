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


class TestAclRefetch:
    """필터 후 top_k 미달 + 후보 페이지가 가득 찬 경우에만 1회 재조회한다."""

    TOP_K = 5

    def _setup(self, monkeypatch, n_chunks, restricted_ids, bm25_enabled):
        """n_chunks개 청크(id 1..n, 낮은 id가 상위 랭크)의 DB와 호출 기록용 검색 mock을 만든다."""
        engine = create_engine("sqlite://")
        DocumentChunk.__table__.create(engine)
        session = Session(engine)
        for cid in range(1, n_chunks + 1):
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
                    sensitivity_level="restricted" if cid in restricted_ids else "normal",
                )
            )
        session.commit()

        hits = [{"chunk_id": cid, "score": 1.0 - cid * 0.001} for cid in range(1, n_chunks + 1)]
        bm25_hits = [BM25SearchResult(chunk_id=cid, bm25_rank=cid - 1) for cid in range(1, n_chunks + 1)]
        calls = {"vector": [], "bm25": [], "embed": 0}

        async def fake_embed(texts):
            calls["embed"] += 1
            return SimpleNamespace(embeddings=[[0.0]])

        def vec_search(emb, workspace_id, top_k):
            calls["vector"].append(top_k)
            return hits[:top_k]

        def bm25_search(q, ws, db, k):
            calls["bm25"].append(k)
            return bm25_hits[:k]

        monkeypatch.setattr(retriever, "embed_texts", fake_embed)
        monkeypatch.setattr(
            retriever, "get_settings", lambda: SimpleNamespace(bm25_enabled=bm25_enabled)
        )
        monkeypatch.setattr(retriever, "get_vector_store", lambda: SimpleNamespace(search=vec_search))
        monkeypatch.setattr(retriever, "get_bm25_manager", lambda: SimpleNamespace(search=bm25_search))
        monkeypatch.setattr(
            retriever, "_resolve_title_and_author", lambda chunk, db: (f"t{chunk.id}", None)
        )
        return session, calls

    @pytest.mark.parametrize("bm25_enabled", [True, False])
    def test_refetch_fills_top_k(self, monkeypatch, bm25_enabled):
        from app.core.metrics import StageTimer

        # 상위 15건(=3배)은 전부 restricted, 이후 15건은 normal
        db, calls = self._setup(monkeypatch, 30, set(range(1, 16)), bm25_enabled)
        timer = StageTimer()
        result = asyncio.run(
            retriever.retrieve("q", "ws", db, top_k=self.TOP_K, access=AccessFilter(), timer=timer)
        )
        assert len(result) == self.TOP_K
        assert all(c.chunk_id > 15 for c in result)
        assert calls["vector"] == [15, 30]
        if bm25_enabled:
            assert calls["bm25"] == [15, 30]
        assert calls["embed"] == 1  # 임베딩 재사용
        assert timer.fields["acl_refetch_count"] == 1

    @pytest.mark.parametrize("bm25_enabled", [True, False])
    def test_no_refetch_when_candidates_not_full(self, monkeypatch, bm25_enabled):
        from app.core.metrics import StageTimer

        # 워크스페이스에 10건뿐(<15): 후보 페이지가 차지 않아 소진된 것으로 본다
        db, calls = self._setup(monkeypatch, 10, set(range(1, 9)), bm25_enabled)
        timer = StageTimer()
        result = asyncio.run(
            retriever.retrieve("q", "ws", db, top_k=self.TOP_K, access=AccessFilter(), timer=timer)
        )
        assert len(result) == 2
        assert len(calls["vector"]) == 1
        assert len(calls["bm25"]) == (1 if bm25_enabled else 0)
        assert timer.fields["acl_refetch_count"] == 0

    @pytest.mark.parametrize("bm25_enabled", [True, False])
    def test_no_refetch_without_access(self, monkeypatch, bm25_enabled):
        db, calls = self._setup(monkeypatch, 30, set(range(1, 16)), bm25_enabled)
        asyncio.run(retriever.retrieve("q", "ws", db, top_k=self.TOP_K))
        assert len(calls["vector"]) == 1

    @pytest.mark.parametrize("bm25_enabled", [True, False])
    def test_no_refetch_when_top_k_already_filled(self, monkeypatch, bm25_enabled):
        db, calls = self._setup(monkeypatch, 30, {1}, bm25_enabled)
        result = asyncio.run(
            retriever.retrieve("q", "ws", db, top_k=self.TOP_K, access=AccessFilter())
        )
        assert len(result) == self.TOP_K
        assert len(calls["vector"]) == 1

    @pytest.mark.parametrize("bm25_enabled", [True, False])
    def test_factor_never_exceeds_cap(self, monkeypatch, bm25_enabled):
        # 재조회 횟수 상한을 풀어도 배수는 12를 넘지 않는다 (3 -> 6 -> 12 -> 중단)
        monkeypatch.setattr(retriever, "_ACL_MAX_REFETCH", 10)
        db, calls = self._setup(monkeypatch, 100, set(range(1, 101)), bm25_enabled)
        result = asyncio.run(
            retriever.retrieve("q", "ws", db, top_k=self.TOP_K, access=AccessFilter())
        )
        assert result == []
        assert calls["vector"] == [15, 30, 60]
        assert max(calls["vector"]) == self.TOP_K * retriever._OVERFETCH_MAX_FACTOR

    @pytest.mark.parametrize("bm25_enabled", [True, False])
    def test_default_refetch_limited_to_once(self, monkeypatch, bm25_enabled):
        db, calls = self._setup(monkeypatch, 100, set(range(1, 101)), bm25_enabled)
        asyncio.run(retriever.retrieve("q", "ws", db, top_k=self.TOP_K, access=AccessFilter()))
        assert calls["vector"] == [15, 30]

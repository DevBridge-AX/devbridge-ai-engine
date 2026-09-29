"""
retriever 관측성 계측 유닛 테스트.

timer(StageTimer)를 retrieve()에 전달하면 embed_ms/bm25_ms/vector_ms/acl_filter_ms
구간이 기록되는지, timer를 넘기지 않으면 기존 동작과 동일한지 검증합니다.
임베딩 API·Chroma·BM25는 mock, DB는 인메모리 SQLite를 사용합니다.
"""

import asyncio
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.metrics import StageTimer
from app.core.rag import retriever
from app.core.rag.bm25_index import BM25SearchResult
from app.core.rag.retriever import AccessFilter
from app.db.models import ChunkSourceType, DocumentChunk


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    DocumentChunk.__table__.create(engine)
    with Session(engine) as session:
        for cid in (1, 2, 3):
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
                )
            )
        session.commit()
        yield session


@pytest.fixture(autouse=True)
def patch_deps(monkeypatch):
    async def fake_embed(texts):
        return SimpleNamespace(embeddings=[[0.0]])

    vector_hits = [{"chunk_id": cid, "score": 0.9 - cid * 0.1} for cid in (1, 2, 3)]
    bm25_hits = [BM25SearchResult(chunk_id=cid, bm25_rank=i) for i, cid in enumerate((2, 3))]

    monkeypatch.setattr(retriever, "embed_texts", fake_embed)
    monkeypatch.setattr(retriever, "get_settings", lambda: SimpleNamespace(bm25_enabled=True))
    monkeypatch.setattr(
        retriever,
        "get_vector_store",
        lambda: SimpleNamespace(search=lambda emb, workspace_id, top_k: vector_hits[:top_k]),
    )
    monkeypatch.setattr(
        retriever, "get_bm25_manager", lambda: SimpleNamespace(search=lambda q, ws, db, k: bm25_hits[:k])
    )
    monkeypatch.setattr(retriever, "_resolve_title_and_author", lambda chunk, db: (f"t{chunk.id}", None))


class TestHybridRetrieveTimer:

    def test_timer_records_stage_keys(self, db):
        timer = StageTimer()
        asyncio.run(retriever.retrieve("q", "ws", db, top_k=3, timer=timer))

        assert "embed_ms" in timer.stages
        assert "bm25_ms" in timer.stages
        assert "vector_ms" in timer.stages
        assert "acl_filter_ms" not in timer.stages  # access 없으면 필터 구간 없음
        assert all(v >= 0.0 for v in timer.stages.values())

    def test_timer_records_acl_filter_ms_when_access_given(self, db):
        timer = StageTimer()
        access = AccessFilter(accessible_task_ids=None, can_view_restricted=False)
        asyncio.run(retriever.retrieve("q", "ws", db, top_k=3, access=access, timer=timer))

        assert "acl_filter_ms" in timer.stages

    def test_no_timer_behaves_like_before(self, db):
        result = asyncio.run(retriever.retrieve("q", "ws", db, top_k=3))
        assert {c.chunk_id for c in result} == {1, 2, 3}


class TestVectorOnlyRetrieveTimer:

    def test_timer_records_embed_and_vector_ms(self, db, monkeypatch):
        monkeypatch.setattr(retriever, "get_settings", lambda: SimpleNamespace(bm25_enabled=False))
        timer = StageTimer()

        asyncio.run(retriever.retrieve("q", "ws", db, top_k=3, timer=timer))

        assert "embed_ms" in timer.stages
        assert "vector_ms" in timer.stages
        assert "acl_filter_ms" not in timer.stages

    def test_no_timer_behaves_like_before(self, db, monkeypatch):
        monkeypatch.setattr(retriever, "get_settings", lambda: SimpleNamespace(bm25_enabled=False))
        result = asyncio.run(retriever.retrieve("q", "ws", db, top_k=3))
        assert {c.chunk_id for c in result} == {1, 2, 3}

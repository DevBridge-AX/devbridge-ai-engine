"""owner_answer_ingestion 성공 시 시맨틱 캐시 무효화 테스트."""

import asyncio
from types import SimpleNamespace

from app.pipelines import owner_answer_ingestion


class _FakeDb:
    def add(self, obj):
        pass

    def flush(self):
        pass


def test_success_invalidates_semantic_cache(monkeypatch):
    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    invalidated: list[str] = []
    monkeypatch.setattr(owner_answer_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(
        owner_answer_ingestion, "get_vector_store", lambda: SimpleNamespace(add=lambda **kw: None)
    )
    monkeypatch.setattr(
        owner_answer_ingestion,
        "get_bm25_manager",
        lambda: SimpleNamespace(invalidate=lambda workspace_id: None),
    )
    monkeypatch.setattr(owner_answer_ingestion, "log_embedding_usage", lambda *a, **kw: None)
    monkeypatch.setattr(
        owner_answer_ingestion,
        "get_semantic_cache",
        lambda: SimpleNamespace(invalidate_workspace=invalidated.append),
    )

    asyncio.run(
        owner_answer_ingestion._run(
            db=_FakeDb(),
            workspace_id="ws-1",
            confirmation_id="c-1",
            question="배포는 어떻게 하나요?",
            answer="main 브랜치에 머지하면 자동 배포됩니다.",
            owner_employee_id="e-1",
            owner_name="홍길동",
        )
    )

    assert invalidated == ["ws-1"]

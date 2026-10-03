"""질의 측 임베딩 taskType 설정·호출부 전달 테스트 (네트워크 호출 없음)."""

import asyncio
import typing
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import EMBEDDING_TASK_TYPES, EmbeddingTaskType, Settings
from app.core import chat_pipeline
from app.core.embeddings import embedder
from app.core.rag import retriever
from app.schemas.chat import ChatRequest


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_default_is_none(monkeypatch):
    monkeypatch.delenv("EMBEDDING_QUERY_TASK_TYPE", raising=False)
    assert _settings().embedding_query_task_type is None


def test_env_override_parses(monkeypatch):
    monkeypatch.setenv("EMBEDDING_QUERY_TASK_TYPE", "SEMANTIC_SIMILARITY")
    assert _settings().embedding_query_task_type == "SEMANTIC_SIMILARITY"


@pytest.mark.parametrize("value", ["semantic_similarity", "NOPE"])
def test_invalid_value_rejected(value):
    with pytest.raises(ValidationError):
        _settings(embedding_query_task_type=value)


def test_task_types_tuple_matches_literal():
    assert set(EMBEDDING_TASK_TYPES) == set(typing.get_args(EmbeddingTaskType))
    assert embedder.EMBEDDING_TASK_TYPES == EMBEDDING_TASK_TYPES


def _set_task_type(monkeypatch, value):
    monkeypatch.setattr(embedder, "get_settings", lambda: SimpleNamespace(embedding_query_task_type=value))


_CASES = [(None, {}), ("SEMANTIC_SIMILARITY", {"task_type": "SEMANTIC_SIMILARITY"})]


@pytest.mark.parametrize("value, expected", _CASES)
def test_retriever_query_embedding_passes_task_type_only_when_set(monkeypatch, value, expected):
    _set_task_type(monkeypatch, value)
    seen = []

    async def fake_embed(texts, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(embeddings=[[0.0]])

    monkeypatch.setattr(retriever, "embed_texts", fake_embed)

    assert asyncio.run(retriever._get_query_embedding("q")) == [0.0]
    assert seen == [expected]


class _Stop(Exception):
    pass


@pytest.mark.parametrize("value, expected", _CASES)
def test_chat_pipeline_embed_passes_task_type_only_when_set(monkeypatch, value, expected):
    _set_task_type(monkeypatch, value)
    seen = []

    async def fake_embed(texts, **kwargs):
        seen.append(kwargs)
        raise _Stop

    monkeypatch.setattr(
        chat_pipeline,
        "get_settings",
        lambda: SimpleNamespace(main_model="m", main_max_tokens=10, semantic_cache_enabled=True),
    )
    monkeypatch.setattr(chat_pipeline, "get_semantic_cache", lambda: SimpleNamespace())
    monkeypatch.setattr(chat_pipeline, "embed_texts", fake_embed)

    request = ChatRequest(
        session_id="s", content="질문", conversation_history=[],
        workspace_id="ws", user_id="u", role="planner",
    )

    async def consume():
        return [e async for e in chat_pipeline.run(request, db=None)]

    with pytest.raises(_Stop):
        asyncio.run(consume())
    assert seen == [expected]

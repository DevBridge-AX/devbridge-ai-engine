"""
chat_pipeline 시맨틱 캐시 통합 유닛 테스트.

검색/그라운딩/LLM/임베딩은 모두 mock이며, 캐시는 테스트마다 새 SemanticCache를 사용합니다.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core import chat_pipeline, metrics
from app.core.cache.semantic_cache import SemanticCache
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest

ANSWER = "안녕하세요. " * 20  # 40자 초과 → 여러 token 조각
THRESHOLD = 0.95


def _request(**overrides) -> ChatRequest:
    payload = {
        "session_id": "s-1",
        "content": "배포 절차 알려줘",
        "conversation_history": [],
        "workspace_id": "ws-1",
        "user_id": "user-1",
        "role": "planner",
        **overrides,
    }
    return ChatRequest(**payload)


def _history():
    return [
        {"role": "user", "content": "이전 질문"},
        {"role": "assistant", "content": "이전 답변"},
    ]


def _chunk(chunk_id=1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id, source_type="document", source_id="doc-1",
        title="제목", content="내용", similarity_score=0.8,
    )


@pytest.fixture
def env(monkeypatch, tmp_path):
    state = SimpleNamespace(
        cache=SemanticCache(threshold=THRESHOLD, ttl_seconds=3600, max_entries=100),
        embed_calls=[],
        retrieve_calls=[],
        retrieve_kwargs=[],
        on_stream=None,
        assess_calls=0,
        stream_calls=0,
        allowed_override=None,
        grounding=GroundingResult(is_groundable=True, confidence=0.9),
        answer_pieces=[ANSWER],
        embedding=[1.0, 0.0],
        settings=SimpleNamespace(
            main_model="fake-main", main_max_tokens=100, semantic_cache_enabled=True
        ),
    )

    async def fake_embed(texts):
        state.embed_calls.append(list(texts))
        return SimpleNamespace(embeddings=[list(state.embedding) for _ in texts])

    async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None, **kwargs):
        state.retrieve_kwargs.append(kwargs)
        state.retrieve_calls.append(kwargs.get("query_embedding"))
        return [_chunk()]

    async def fake_assess(chunks, query):
        state.assess_calls += 1
        return state.grounding

    async def fake_stream(messages, system_prompt, max_tokens=4096):
        state.stream_calls += 1
        if state.on_stream is not None:
            state.on_stream()
        for piece in state.answer_pieces:
            yield piece, None
        yield None, LLMUsage(model="claude-main", prompt_tokens=100, completion_tokens=20)

    async def fake_rewrite(content, history):
        return content, LLMUsage(model="haiku", prompt_tokens=7, completion_tokens=3)

    monkeypatch.setattr(chat_pipeline, "get_settings", lambda: state.settings)
    monkeypatch.setattr(chat_pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(chat_pipeline, "get_semantic_cache", lambda: state.cache)
    monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
    monkeypatch.setattr(chat_pipeline.llm, "call_main_stream", fake_stream)
    monkeypatch.setattr(chat_pipeline.query_rewriter, "rewrite", fake_rewrite)
    monkeypatch.setattr(
        chat_pipeline.retriever,
        "_allowed_chunk_ids",
        lambda ids, access, db: set(ids) if state.allowed_override is None else state.allowed_override,
    )
    monkeypatch.setattr(
        metrics,
        "get_settings",
        lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(tmp_path)),
    )
    state.metrics_dir = tmp_path
    return state


def _run(request):
    async def consume():
        return [e async for e in chat_pipeline.run(request, db=None)]

    return asyncio.run(consume())


def _records(env):
    path = env.metrics_dir / "chat_metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestFlagOff:

    def test_no_embed_no_store(self, env):
        env.settings.semantic_cache_enabled = False

        events = _run(_request())

        assert env.embed_calls == []
        assert env.retrieve_kwargs == [{}]  # query_embedding 키 자체가 전달되지 않는다
        assert env.cache.size() == 0
        assert events[-1].event == "done"
        record = _records(env)[0]
        assert (record["cache_enabled"], record["cache_hit"]) == (False, False)
        assert record["cache_similarity"] is None and record["cache_lookup_ms"] is None


class TestMissThenHit:

    def test_miss_stores_then_second_request_hits(self, env):
        first = _run(_request())

        assert len(env.embed_calls) == 1
        assert env.retrieve_calls == [[1.0, 0.0]]  # 파이프라인이 계산한 임베딩 재사용
        assert env.cache.size() == 1
        first_done = first[-1].data

        second = _run(_request())

        assert (env.assess_calls, env.stream_calls, len(env.retrieve_calls)) == (1, 1, 1)
        assert [e.event for e in second] == ["token"] * (len(second) - 1) + ["done"]
        assert len(second) - 1 > 1
        assert "".join(e.data["text"] for e in second[:-1]) == ANSWER
        done = second[-1].data
        assert done["is_groundable"] is True
        assert done["confidence"] == first_done["confidence"]
        assert done["citations"] == first_done["citations"]
        assert done["suggested_owner_id"] is None
        assert done["token_usage"]["main"]["prompt_tokens"] == 0
        assert done["token_usage"]["main"]["completion_tokens"] == 0
        assert done["token_usage"]["grounding"] is None

        record = _records(env)[-1]
        assert record["cache_enabled"] is True
        assert record["cache_hit"] is True
        assert record["grounding_stage"] == "cache"
        assert record["cache_similarity"] >= THRESHOLD
        assert record["cache_lookup_ms"] >= 0
        assert record["ttft_ms"] is not None
        assert ANSWER not in json.dumps(record, ensure_ascii=False)

    def test_miss_record_has_cache_fields(self, env):
        _run(_request())
        record = _records(env)[0]
        assert record["cache_enabled"] is True
        assert record["cache_hit"] is False
        assert record["cache_similarity"] is None
        assert record["cache_lookup_ms"] >= 0

    def test_dissimilar_query_misses(self, env):
        _run(_request())
        env.embedding = [0.0, 1.0]
        _run(_request(content="전혀 다른 질문"))
        assert env.stream_calls == 2
        assert env.cache.size() == 2


class TestStoreConditions:

    def test_turn2_looks_up_but_does_not_store(self, env):
        events = _run(_request(conversation_history=_history()))

        assert len(env.embed_calls) == 1
        assert env.cache.size() == 0
        assert events[-1].data["is_groundable"] is True

    def test_fallback_reason_not_stored(self, env):
        env.grounding = GroundingResult(is_groundable=True, confidence=0.5, fallback_reason="llm_error")
        _run(_request())
        assert env.cache.size() == 0

    def test_not_groundable_not_stored(self, env):
        env.grounding = GroundingResult(is_groundable=False, confidence=0.1)
        _run(_request())
        assert env.cache.size() == 0
        assert env.stream_calls == 0

    def test_invalidation_during_stream_skips_store(self, env):
        env.on_stream = lambda: env.cache.invalidate_workspace("ws-1")

        events = _run(_request())

        assert events[-1].event == "done"
        assert env.cache.size() == 0

    def test_invalidation_of_other_workspace_still_stores(self, env):
        env.on_stream = lambda: env.cache.invalidate_workspace("other-ws")
        _run(_request())
        assert env.cache.size() == 1

    def test_empty_answer_not_stored(self, env):
        env.answer_pieces = ["  ", ""]
        _run(_request())
        assert env.cache.size() == 0


class TestHitSafeguards:

    def test_access_recheck_failure_falls_back_to_miss(self, env):
        _run(_request())
        assert env.cache.size() == 1

        env.allowed_override = set()  # 캐시된 근거 청크가 더 이상 허용되지 않음
        events = _run(_request())

        assert env.stream_calls == 2
        assert len(env.retrieve_calls) == 2
        assert events[-1].data["token_usage"]["main"]["completion_tokens"] == 20
        # 실패한 엔트리는 제거되고 새 답변이 다시 저장된다
        assert env.cache.size() == 1
        assert _records(env)[-1]["cache_hit"] is False

    def test_rewrite_usage_passed_through_on_hit(self, env):
        env.cache.clear()
        _run(_request())  # 첫 턴 저장 (임베딩 [1, 0], 네임스페이스 동일)
        events = _run(_request(conversation_history=_history()))

        assert env.stream_calls == 1  # turn 2도 hit
        rewrite = events[-1].data["token_usage"]["rewrite"]
        assert rewrite == {"model": "haiku", "prompt_tokens": 7, "completion_tokens": 3}
        assert _records(env)[-1]["rewrite_prompt_tokens"] == 7

    def test_different_access_scope_does_not_hit(self, env):
        _run(_request(accessible_task_ids=["t1"]))
        _run(_request(accessible_task_ids=[]))
        _run(_request(accessible_task_ids=None))
        assert env.stream_calls == 3

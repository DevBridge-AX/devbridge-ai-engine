"""
시맨틱 캐시 LLM 재검증(verify) 테스트.

LLM/임베딩/검색은 전부 mock이며 네트워크 호출이 없습니다.
"""

import asyncio
import json
import logging
import math
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core import chat_pipeline, metrics
from app.core.cache import verifier
from app.core.cache.semantic_cache import CacheEntry, SemanticCache, make_namespace
from app.core.cache.verifier import VerifyResult, verify_same_question
from app.core.llm import provider
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest

THRESHOLD = 0.95
CANDIDATE = 0.86
ANSWER = "안녕하세요. " * 20


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class TestConfig:

    def test_defaults(self):
        s = Settings(_env_file=None)
        assert s.semantic_cache_verify_enabled is False
        assert s.semantic_cache_candidate_threshold == 0.86
        assert s.semantic_cache_verify_timeout_seconds == 3.0
        assert s.semantic_cache_enabled is False
        assert s.semantic_cache_threshold == 0.95

    @pytest.mark.parametrize("value", [-0.1, 1.1])
    def test_candidate_threshold_range(self, value):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, semantic_cache_candidate_threshold=value)

    @pytest.mark.parametrize("value", [0, -1])
    def test_timeout_must_be_positive(self, value):
        with pytest.raises(ValidationError):
            Settings(_env_file=None, semantic_cache_verify_timeout_seconds=value)


# ---------------------------------------------------------------------------
# call_rewrite purpose 전파
# ---------------------------------------------------------------------------

class TestCallRewritePurpose:

    @staticmethod
    def _patch(monkeypatch):
        recorded = []

        async def fake_request(url, headers, body, timeout):
            return {}

        monkeypatch.setattr(provider, "_request_once", fake_request)
        monkeypatch.setattr(
            provider,
            "_parse_response",
            lambda p, d: ("YES", {"input_tokens": 5, "output_tokens": 1, "thoughts_tokens": 0, "finish_reason": "stop"}),
        )
        monkeypatch.setattr(provider, "_record_llm_call", lambda **kw: recorded.append(kw))
        monkeypatch.setattr(
            provider,
            "get_settings",
            lambda: SimpleNamespace(
                rewrite_model="claude-haiku-x", anthropic_base_url="http://x", gms_api_key="k",
                gms_base_url="http://x", gemini_base_url="http://x", openai_base_url="http://x",
            ),
            raising=False,
        )
        monkeypatch.setattr(provider, "_build_url", lambda *a, **k: "http://x")
        monkeypatch.setattr(provider, "_build_headers", lambda *a, **k: {})
        return recorded

    def test_default_purpose_is_rewrite(self, monkeypatch):
        recorded = self._patch(monkeypatch)
        asyncio.run(provider.call_rewrite([{"role": "user", "content": "q"}], "sys"))
        assert recorded[0]["purpose"] == "rewrite"

    def test_custom_purpose(self, monkeypatch):
        recorded = self._patch(monkeypatch)
        text, usage = asyncio.run(
            provider.call_rewrite([{"role": "user", "content": "q"}], "sys", max_tokens=8, purpose="cache_verify")
        )
        assert recorded[0]["purpose"] == "cache_verify"
        assert (text, usage.prompt_tokens, usage.completion_tokens) == ("YES", 5, 1)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

def _patch_verifier(monkeypatch, *, text=None, exc=None, delay=0.0, timeout=3.0):
    calls = []

    async def fake_call_rewrite(messages, system_prompt="", max_tokens=512, purpose="rewrite"):
        calls.append({"messages": messages, "system": system_prompt, "max_tokens": max_tokens, "purpose": purpose})
        if delay:
            await asyncio.sleep(delay)
        if exc is not None:
            raise exc
        return text, LLMUsage(model="haiku", prompt_tokens=30, completion_tokens=1)

    monkeypatch.setattr(verifier.llm, "call_rewrite", fake_call_rewrite)
    monkeypatch.setattr(
        verifier, "get_settings", lambda: SimpleNamespace(semantic_cache_verify_timeout_seconds=timeout)
    )
    return calls


class TestVerifier:

    @pytest.mark.parametrize("text", ["YES", "yes", " Yes. ", "YES!", "`YES`"])
    def test_yes_variants(self, monkeypatch, text):
        calls = _patch_verifier(monkeypatch, text=text)
        result = asyncio.run(verify_same_question("새 질문", "캐시 질문"))
        assert (result.same, result.outcome) == (True, "yes")
        assert result.usage.prompt_tokens == 30
        assert result.latency_ms >= 0
        assert calls[0]["purpose"] == "cache_verify"
        assert calls[0]["max_tokens"] <= 16
        assert verifier.CACHE_VERIFY_PROMPT_VERSION == "v1"

    @pytest.mark.parametrize("text", ["NO", "no.", "No"])
    def test_no(self, monkeypatch, text):
        _patch_verifier(monkeypatch, text=text)
        result = asyncio.run(verify_same_question("a", "b"))
        assert (result.same, result.outcome) == (False, "no")

    @pytest.mark.parametrize("text", ["", "YES, 같습니다", "아마도", "NOT SURE", "YES NO"])
    def test_invalid_is_fail_closed(self, monkeypatch, text):
        _patch_verifier(monkeypatch, text=text)
        result = asyncio.run(verify_same_question("a", "b"))
        assert (result.same, result.outcome) == (False, "invalid")

    def test_exception_is_fail_closed(self, monkeypatch):
        _patch_verifier(monkeypatch, exc=RuntimeError("boom"))
        result = asyncio.run(verify_same_question("a", "b"))
        assert (result.same, result.outcome, result.usage) == (False, "error", None)

    def test_timeout_is_fail_closed(self, monkeypatch):
        _patch_verifier(monkeypatch, text="YES", delay=0.5, timeout=0.01)
        result = asyncio.run(verify_same_question("a", "b"))
        assert (result.same, result.outcome) == (False, "timeout")

    def test_question_text_not_logged(self, monkeypatch, caplog):
        _patch_verifier(monkeypatch, exc=RuntimeError("secret-question 포함"))
        with caplog.at_level(logging.WARNING):
            asyncio.run(verify_same_question("비밀질문A", "비밀질문B"))
        assert "비밀질문" not in caplog.text

    def test_prompt_contains_both_questions(self, monkeypatch):
        calls = _patch_verifier(monkeypatch, text="NO")
        asyncio.run(verify_same_question("새로운질문", "캐시된질문"))
        content = calls[0]["messages"][0]["content"]
        assert "새로운질문" in content and "캐시된질문" in content

    def test_prompt_has_no_dataset_leakage(self):
        from pathlib import Path

        dataset = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "datasets" / "cache_pairs.jsonl"
        for line in dataset.read_text(encoding="utf-8").splitlines():
            if line.strip():
                pair = json.loads(line)
                assert pair["q1"] not in verifier.CACHE_VERIFY_SYSTEM_PROMPT
                assert pair["q2"] not in verifier.CACHE_VERIFY_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# SemanticCache.lookup(min_similarity) / mark_hit
# ---------------------------------------------------------------------------

_NS = make_namespace("ws-1", "planner", None, False, "pv", "m")


def _entry(query="q"):
    return CacheEntry(
        embedding=[], query=query, answer="a", citations=[], confidence=0.9,
        suggested_owner_id=None, created_at=0.0,
    )


def _vec(cos):
    return [cos, math.sqrt(1 - cos * cos)]


class TestLookupMinSimilarity:

    def _cache(self):
        cache = SemanticCache(threshold=THRESHOLD, ttl_seconds=3600, max_entries=100, clock=lambda: 0.0)
        return cache

    def test_default_unchanged_below_threshold_misses(self):
        cache = self._cache()
        cache.store(_NS, [1.0, 0.0], _entry())
        assert cache.lookup(_NS, _vec(0.9)) is None

    def test_candidate_returned_without_side_effects(self):
        cache = self._cache()
        first, second = _entry("first"), _entry("second")
        cache.store(_NS, [1.0, 0.0], first)
        cache.store(_NS, [0.0, 1.0], second)

        found = cache.lookup(_NS, _vec(0.9), min_similarity=CANDIDATE)

        assert found is not None and found[0] is first
        assert found[1] == pytest.approx(0.9)
        assert first.hits == 0
        assert next(iter(cache._store[_NS].values())) is first  # LRU 순서 불변

    def test_mark_hit_bumps_hits_and_lru(self):
        cache = self._cache()
        first, second = _entry("first"), _entry("second")
        cache.store(_NS, [1.0, 0.0], first)
        cache.store(_NS, [0.0, 1.0], second)
        entry, _ = cache.lookup(_NS, _vec(0.9), min_similarity=CANDIDATE)

        cache.mark_hit(_NS, entry)

        assert first.hits == 1
        assert list(cache._store[_NS].values())[-1] is first

    def test_mark_hit_on_removed_entry_is_noop(self):
        cache = self._cache()
        entry = _entry()
        cache.store(_NS, [1.0, 0.0], entry)
        cache.remove(_NS, entry)
        cache.mark_hit(_NS, entry)
        assert entry.hits == 0

    def test_at_or_above_threshold_bumps_even_with_floor(self):
        cache = self._cache()
        entry = _entry()
        cache.store(_NS, [1.0, 0.0], entry)
        found = cache.lookup(_NS, [1.0, 0.0], min_similarity=CANDIDATE)
        assert found[0].hits == 1

    def test_below_candidate_floor_misses(self):
        cache = self._cache()
        cache.store(_NS, [1.0, 0.0], _entry())
        assert cache.lookup(_NS, _vec(0.8), min_similarity=CANDIDATE) is None


# ---------------------------------------------------------------------------
# 파이프라인
# ---------------------------------------------------------------------------

def _request(**overrides) -> ChatRequest:
    payload = {
        "session_id": "s-1", "content": "배포 절차 알려줘", "conversation_history": [],
        "workspace_id": "ws-1", "user_id": "user-1", "role": "planner", **overrides,
    }
    return ChatRequest(**payload)


@pytest.fixture
def env(monkeypatch, tmp_path):
    state = SimpleNamespace(
        cache=SemanticCache(threshold=THRESHOLD, ttl_seconds=3600, max_entries=100),
        embedding=[1.0, 0.0],
        stream_calls=0,
        verify_calls=[],
        verdict=VerifyResult(same=True, outcome="yes", latency_ms=1.0),
        allowed_override=None,
        settings=SimpleNamespace(
            main_model="fake-main", main_max_tokens=100, semantic_cache_enabled=True,
            semantic_cache_verify_enabled=True, semantic_cache_candidate_threshold=CANDIDATE,
        ),
    )

    async def fake_embed(texts):
        return SimpleNamespace(embeddings=[list(state.embedding) for _ in texts])

    async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None, **kwargs):
        return [RetrievedChunk(chunk_id=1, source_type="document", source_id="d", title="t",
                               content="c", similarity_score=0.8)]

    async def fake_assess(chunks, query):
        return GroundingResult(is_groundable=True, confidence=0.9)

    async def fake_stream(messages, system_prompt, max_tokens=4096):
        state.stream_calls += 1
        yield ANSWER, None
        yield None, LLMUsage(model="claude-main", prompt_tokens=100, completion_tokens=20)

    async def fake_verify(new_query, cached_query):
        state.verify_calls.append((new_query, cached_query))
        return state.verdict

    monkeypatch.setattr(chat_pipeline, "get_settings", lambda: state.settings)
    monkeypatch.setattr(chat_pipeline, "embed_texts", fake_embed)
    monkeypatch.setattr(chat_pipeline, "get_semantic_cache", lambda: state.cache)
    monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
    monkeypatch.setattr(chat_pipeline.llm, "call_main_stream", fake_stream)
    monkeypatch.setattr(chat_pipeline.verifier, "verify_same_question", fake_verify)
    monkeypatch.setattr(
        chat_pipeline.retriever, "_allowed_chunk_ids",
        lambda ids, access, db: set(ids) if state.allowed_override is None else state.allowed_override,
    )
    monkeypatch.setattr(
        metrics, "get_settings", lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(tmp_path))
    )
    state.metrics_dir = tmp_path
    return state


def _run(request):
    async def consume():
        return [e async for e in chat_pipeline.run(request, db=None)]

    return asyncio.run(consume())


def _last_record(env):
    path = env.metrics_dir / "chat_metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line][-1]


def _seed(env):
    """첫 요청으로 캐시에 엔트리를 채우고 stream 카운터를 초기화합니다."""
    _run(_request())
    assert env.cache.size() == 1
    env.stream_calls = 0


class TestPipelineVerify:

    def test_verify_disabled_candidate_band_is_plain_miss(self, env):
        env.settings.semantic_cache_verify_enabled = False
        _seed(env)
        env.embedding = _vec(0.9)

        _run(_request())

        assert env.verify_calls == []
        assert env.stream_calls == 1

    def test_missing_verify_settings_behave_as_disabled(self, env):
        env.settings = SimpleNamespace(main_model="fake-main", main_max_tokens=100, semantic_cache_enabled=True)
        _seed(env)
        env.embedding = _vec(0.9)
        _run(_request())
        assert env.verify_calls == [] and env.stream_calls == 1

    def test_above_threshold_hits_without_verify(self, env):
        _seed(env)
        events = _run(_request())  # 동일 임베딩 → 유사도 1.0
        assert env.verify_calls == []
        assert env.stream_calls == 0
        assert events[-1].data["token_usage"]["main"]["prompt_tokens"] == 0

    def test_candidate_yes_hits(self, env):
        _seed(env)
        env.embedding = _vec(0.9)

        events = _run(_request(content="다른 표현의 질문"))

        assert env.verify_calls == [("다른 표현의 질문", "배포 절차 알려줘")]
        assert env.stream_calls == 0
        done = events[-1].data
        assert done["token_usage"]["main"] == {"model": "fake-main", "prompt_tokens": 0, "completion_tokens": 0}
        assert done["token_usage"]["grounding"] is None
        record = _last_record(env)
        assert record["grounding_stage"] == "cache"
        assert record["cache_hit"] is True
        assert record["cache_similarity"] == pytest.approx(0.9)
        entry = next(iter(env.cache._store[next(iter(env.cache._store))].values()))
        assert entry.hits == 1

    def test_candidate_no_runs_main_llm(self, env):
        _seed(env)
        env.verdict = VerifyResult(same=False, outcome="no", latency_ms=1.0)
        env.embedding = _vec(0.9)

        events = _run(_request())

        assert len(env.verify_calls) == 1
        assert env.stream_calls == 1
        assert events[-1].data["token_usage"]["main"]["completion_tokens"] == 20
        record = _last_record(env)
        assert record["cache_hit"] is False and record["grounding_stage"] != "cache"
        # 검증 실패한 후보의 hits는 올라가지 않는다
        first_entry = next(iter(next(iter(env.cache._store.values())).values()))
        assert first_entry.hits == 0

    def test_verifier_error_is_miss(self, env):
        _seed(env)
        env.verdict = VerifyResult(same=False, outcome="error", latency_ms=1.0)
        env.embedding = _vec(0.9)
        _run(_request())
        assert env.stream_calls == 1

    def test_below_candidate_threshold_skips_verify(self, env):
        _seed(env)
        env.embedding = _vec(0.8)
        _run(_request())
        assert env.verify_calls == [] and env.stream_calls == 1

    def test_access_recheck_failure_skips_verify(self, env):
        _seed(env)
        env.allowed_override = set()
        env.embedding = _vec(0.9)

        _run(_request())

        assert env.verify_calls == []
        assert env.stream_calls == 1

    def test_candidate_above_threshold_config_disables_band(self, env):
        env.settings.semantic_cache_candidate_threshold = 0.99
        _seed(env)
        env.embedding = _vec(0.97)  # threshold(0.95) 이상 → 즉시 hit, 검증 없음

        _run(_request())
        assert env.verify_calls == [] and env.stream_calls == 0

        env.embedding = _vec(0.9)  # 후보 구간 없음 → 검증 없이 miss
        _run(_request())
        assert env.verify_calls == [] and env.stream_calls == 1


class TestPipelineVerifyMetrics:
    """chat_metrics의 cache_verify_ms / cache_verify_result 기록."""

    def test_verified_hit_records_yes_and_latency(self, env):
        env.verdict = VerifyResult(same=True, outcome="yes", latency_ms=12.5)
        _seed(env)
        env.embedding = _vec(0.9)

        _run(_request())

        record = _last_record(env)
        assert record["cache_hit"] is True
        assert record["cache_verify_result"] == "yes"
        assert record["cache_verify_ms"] == pytest.approx(12.5)

    def test_rejected_candidate_records_no_on_miss_path(self, env):
        _seed(env)
        env.verdict = VerifyResult(same=False, outcome="no", latency_ms=7.0)
        env.embedding = _vec(0.9)

        _run(_request())

        record = _last_record(env)
        assert record["cache_hit"] is False
        assert record["cache_verify_result"] == "no"
        assert record["cache_verify_ms"] == pytest.approx(7.0)

    def test_timeout_outcome_recorded(self, env):
        _seed(env)
        env.verdict = VerifyResult(same=False, outcome="timeout", latency_ms=3000.0)
        env.embedding = _vec(0.9)

        _run(_request())

        record = _last_record(env)
        assert record["cache_hit"] is False
        assert record["cache_verify_result"] == "timeout"

    def test_verify_disabled_records_none(self, env):
        env.settings.semantic_cache_verify_enabled = False
        _seed(env)
        env.embedding = _vec(0.9)

        _run(_request())

        record = _last_record(env)
        assert record["cache_verify_result"] is None
        assert record["cache_verify_ms"] is None

    def test_above_threshold_hit_has_no_verify(self, env):
        _seed(env)

        _run(_request())

        record = _last_record(env)
        assert record["cache_hit"] is True
        assert record["cache_verify_result"] is None
        assert record["cache_verify_ms"] is None

    def test_access_check_failure_records_none(self, env):
        _seed(env)
        env.allowed_override = set()
        env.embedding = _vec(0.9)

        _run(_request())

        assert _last_record(env)["cache_verify_result"] is None

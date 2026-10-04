"""
시맨틱 캐시 재검증 운영 가드 유닛 테스트: 후보 구간 관계 경고(1회)와 일일 호출 상한.

검색/그라운딩/LLM/임베딩/검증기는 모두 mock입니다.
"""

import asyncio
import datetime
import json
import logging
import math
from types import SimpleNamespace

import pytest

from app.config import Settings
from app.core import chat_pipeline, metrics
from app.core.cache import verifier
from app.core.cache.semantic_cache import SemanticCache
from app.core.cache.verifier import VerifyBudget, VerifyResult, check_verify_band
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest

THRESHOLD = 0.95
CANDIDATE = 0.86
ANSWER = "안녕하세요. " * 20


def _vec(cos):
    return [cos, math.sqrt(1 - cos * cos)]


@pytest.fixture(autouse=True)
def _reset_guards():
    verifier.reset_verify_guards()
    yield
    verifier.reset_verify_guards()


class TestCheckVerifyBand:

    def test_candidate_above_threshold_warns(self):
        assert check_verify_band(0.95, 0.97) is not None

    def test_candidate_equal_threshold_warns(self):
        assert check_verify_band(0.95, 0.95) is not None

    def test_candidate_below_threshold_is_valid(self):
        assert check_verify_band(0.95, 0.86) is None

    def test_warning_logged_once_per_process(self, caplog):
        with caplog.at_level(logging.WARNING, logger=verifier.logger.name):
            for _ in range(3):
                verifier.warn_verify_band_once(0.95, 0.97)
        assert len([r for r in caplog.records if "never runs" in r.getMessage()]) == 1

    def test_valid_band_does_not_log(self, caplog):
        with caplog.at_level(logging.WARNING, logger=verifier.logger.name):
            verifier.warn_verify_band_once(0.95, 0.86)
        assert caplog.records == []

    def test_reset_allows_warning_again(self, caplog):
        with caplog.at_level(logging.WARNING, logger=verifier.logger.name):
            verifier.warn_verify_band_once(0.95, 0.97)
            verifier.reset_verify_guards()
            verifier.warn_verify_band_once(0.95, 0.97)
        assert len(caplog.records) == 2


class TestVerifyBudget:

    def test_limit_zero_is_unlimited(self):
        budget = VerifyBudget()
        assert all(budget.try_acquire(0) for _ in range(100))

    def test_limit_n_allows_n_then_denies(self):
        budget = VerifyBudget(today=lambda: datetime.date(2026, 1, 1))
        assert [budget.try_acquire(2) for _ in range(4)] == [True, True, False, False]

    def test_resets_when_date_changes(self):
        day = {"d": datetime.date(2026, 1, 1)}
        budget = VerifyBudget(today=lambda: day["d"])
        assert budget.try_acquire(1) is True
        assert budget.try_acquire(1) is False
        day["d"] = datetime.date(2026, 1, 2)
        assert budget.try_acquire(1) is True

    def test_exhaustion_warning_once_per_day(self, caplog):
        day = {"d": datetime.date(2026, 1, 1)}
        budget = VerifyBudget(today=lambda: day["d"])
        with caplog.at_level(logging.WARNING, logger=verifier.logger.name):
            for _ in range(4):
                budget.try_acquire(1)
            assert len(caplog.records) == 1
            day["d"] = datetime.date(2026, 1, 2)
            for _ in range(3):
                budget.try_acquire(1)
            assert len(caplog.records) == 2

    def test_singleton_resettable(self):
        first = verifier.get_verify_budget()
        assert verifier.get_verify_budget() is first
        verifier.reset_verify_guards()
        assert verifier.get_verify_budget() is not first

    def test_setting_default_unlimited_and_validated(self):
        assert Settings.model_fields["semantic_cache_verify_daily_limit"].default == 0
        with pytest.raises(ValueError):
            Settings(semantic_cache_verify_daily_limit=-1)


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
        settings=SimpleNamespace(
            main_model="fake-main", main_max_tokens=100, semantic_cache_enabled=True,
            semantic_cache_verify_enabled=True, semantic_cache_candidate_threshold=CANDIDATE,
            semantic_cache_verify_daily_limit=1,
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
        chat_pipeline.retriever, "_allowed_chunk_ids", lambda ids, access, db: set(ids)
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


def _records(env):
    path = env.metrics_dir / "chat_metrics.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _seed(env):
    _run(_request())
    assert env.cache.size() == 1
    env.stream_calls = 0


class TestPipelineDailyLimit:

    def test_second_candidate_request_skips_verify(self, env):
        env.verdict = VerifyResult(same=False, outcome="no", latency_ms=1.0)
        _seed(env)
        env.embedding = _vec(0.9)

        _run(_request())
        # 첫 요청의 miss 답변이 (0.9, +) 방향으로 저장되므로, 반대 방향 임베딩으로 시드(1,0)와만 후보 구간을 만든다.
        env.embedding = [0.9, -math.sqrt(1 - 0.81)]
        events = _run(_request())

        assert len(env.verify_calls) == 1
        assert env.stream_calls == 2  # 두 번 모두 일반 경로
        assert events[-1].event == "done"
        assert events[-1].data["token_usage"]["main"]["completion_tokens"] == 20
        first, second = _records(env)[-2:]
        assert first["cache_verify_result"] == "no"
        assert second["cache_verify_result"] == "limit"
        assert second["cache_verify_ms"] is None
        assert second["cache_hit"] is False

    def test_exhausted_budget_does_not_hit_even_if_verdict_would_be_yes(self, env):
        _seed(env)
        env.embedding = _vec(0.9)
        _run(_request())  # YES hit, 예산 소진
        env.stream_calls = 0

        _run(_request())

        assert len(env.verify_calls) == 1
        assert env.stream_calls == 1
        assert _records(env)[-1]["cache_verify_result"] == "limit"

    def test_immediate_hit_unaffected_by_exhausted_budget(self, env):
        _seed(env)
        verifier.get_verify_budget().try_acquire(1)  # 소진 상태로 만든다
        events = _run(_request())  # 유사도 1.0 → 즉시 hit

        assert env.verify_calls == []
        assert env.stream_calls == 0
        assert events[-1].data["token_usage"]["main"]["prompt_tokens"] == 0
        assert _records(env)[-1]["cache_verify_result"] is None

    def test_limit_zero_is_unlimited(self, env):
        env.settings.semantic_cache_verify_daily_limit = 0
        _seed(env)
        env.embedding = _vec(0.9)
        for _ in range(3):
            _run(_request())
        assert len(env.verify_calls) == 3

    def test_missing_limit_setting_behaves_as_unlimited(self, env):
        del env.settings.semantic_cache_verify_daily_limit
        _seed(env)
        env.embedding = _vec(0.9)
        _run(_request())
        _run(_request())
        assert len(env.verify_calls) == 2

    def test_empty_band_warning_logged_once_across_requests(self, env, caplog):
        env.settings.semantic_cache_candidate_threshold = 0.97
        _seed(env)
        with caplog.at_level(logging.WARNING, logger=verifier.logger.name):
            for _ in range(3):
                _run(_request())
        assert len([r for r in caplog.records if "never runs" in r.getMessage()]) == 1

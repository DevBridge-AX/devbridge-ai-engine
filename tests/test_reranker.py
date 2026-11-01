"""
app/core/rag/reranker.py 유닛 테스트. call_with_model은 mock으로 대체합니다(네트워크 호출 없음).
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.llm.provider import LLMUsage
from app.core.rag import reranker
from app.core.rag.reranker import parse_ranking, rerank
from app.core.rag.retriever import RetrievedChunk


def _chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=i,
            source_type="document",
            source_id=f"doc-{i}",
            title=f"문서{i}",
            content=f"내용{i}",
            similarity_score=round(0.9 - i * 0.05, 2),
        )
        for i in range(1, n + 1)
    ]


def _settings(**overrides):
    base = dict(
        rerank_enabled=True,
        rerank_model="gemini-test",
        rerank_candidates=10,
        rerank_timeout_seconds=2.0,
        rerank_max_chunk_chars=600,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture
def enabled(monkeypatch):
    monkeypatch.setattr(reranker, "get_settings", lambda: _settings())


class TestParseRanking:

    def test_object_form_one_based_to_zero_based(self):
        assert parse_ranking('{"ranking": [3, 1, 2]}', 5, 3) == [2, 0, 1]

    def test_bare_array_accepted(self):
        assert parse_ranking("[2, 1]", 4, 2) == [1, 0]

    def test_out_of_range_and_duplicates_dropped(self):
        assert parse_ranking('{"ranking": [9, 2, 2, 0, 1, -1]}', 4, 2) == [1, 0]

    def test_short_result_filled_in_original_order(self):
        assert parse_ranking('{"ranking": [4]}', 5, 3) == [3, 0, 1]

    def test_truncated_to_top_k(self):
        assert parse_ranking('{"ranking": [5, 4, 3, 2, 1]}', 5, 3) == [4, 3, 2]

    @pytest.mark.parametrize(
        "text", ["not json", "", '{"other": [1]}', '"x"', '{"ranking": "1,2"}', '{"ranking": [99]}']
    )
    def test_unparseable_returns_none(self, text):
        assert parse_ranking(text, 5, 3) is None


class TestRerank:

    async def test_disabled_returns_original_without_call(self, monkeypatch):
        monkeypatch.setattr(reranker, "get_settings", lambda: _settings(rerank_enabled=False))

        async def boom(*a, **k):
            raise AssertionError("호출되면 안 됨")

        monkeypatch.setattr(reranker, "call_with_model", boom)
        chunks = _chunks(8)
        result = await rerank("q", chunks, top_k=5)

        assert result.stage == "disabled"
        assert not result.applied
        assert result.chunks == chunks[:5]
        assert result.usage is None

    async def test_skipped_few_without_call(self, enabled, monkeypatch):
        async def boom(*a, **k):
            raise AssertionError("호출되면 안 됨")

        monkeypatch.setattr(reranker, "call_with_model", boom)
        chunks = _chunks(5)
        result = await rerank("q", chunks, top_k=5)

        assert result.stage == "skipped_few"
        assert result.chunks == chunks

    async def test_applied_reorders_and_keeps_similarity(self, enabled, monkeypatch):
        captured = {}

        async def fake(model, messages, system, **kwargs):
            captured.update(model=model, messages=messages, kwargs=kwargs)
            return json.dumps({"ranking": [7, 2, 5, 1, 3]}), LLMUsage("gemini-test", 100, 9)

        monkeypatch.setattr(reranker, "call_with_model", fake)
        chunks = _chunks(8)
        originals = {c.chunk_id: c.similarity_score for c in chunks}
        result = await rerank("질문", chunks, top_k=5)

        assert result.stage == "applied" and result.applied
        assert [c.chunk_id for c in result.chunks] == [7, 2, 5, 1, 3]
        # 원래 객체 그대로, 점수 불변
        assert all(c.similarity_score == originals[c.chunk_id] for c in result.chunks)
        assert result.chunks[0] is chunks[6]
        assert result.usage.prompt_tokens == 100
        assert captured["model"] == "gemini-test"
        assert captured["kwargs"]["purpose"] == "rerank"
        assert "질문" in captured["messages"][0]["content"]

    async def test_timeout_fallback(self, monkeypatch):
        monkeypatch.setattr(reranker, "get_settings", lambda: _settings(rerank_timeout_seconds=0.01))

        async def slow(*a, **k):
            await asyncio.sleep(1)

        monkeypatch.setattr(reranker, "call_with_model", slow)
        chunks = _chunks(8)
        result = await rerank("q", chunks, top_k=5)

        assert result.stage == "fallback:timeout"
        assert result.chunks == chunks[:5]

    async def test_http_error_fallback(self, enabled, monkeypatch):
        async def fail(*a, **k):
            raise httpx.ConnectError("down")

        monkeypatch.setattr(reranker, "call_with_model", fail)
        chunks = _chunks(8)
        result = await rerank("q", chunks, top_k=5)

        assert result.stage == "fallback:ConnectError"
        assert result.chunks == chunks[:5]

    async def test_http_status_error_fallback_records_status(self, enabled, monkeypatch):
        async def fail(*a, **k):
            request = httpx.Request("POST", "https://example.invalid")
            response = httpx.Response(429, request=request)
            raise httpx.HTTPStatusError("rate limited", request=request, response=response)

        monkeypatch.setattr(reranker, "call_with_model", fail)
        chunks = _chunks(8)
        result = await rerank("q", chunks, top_k=5)

        assert result.stage == "fallback:http_429"
        assert result.chunks == chunks[:5]

    async def test_parse_failure_fallback_keeps_usage(self, enabled, monkeypatch):
        async def fake(*a, **k):
            return "죄송합니다", LLMUsage("gemini-test", 50, 3)

        monkeypatch.setattr(reranker, "call_with_model", fake)
        chunks = _chunks(8)
        result = await rerank("q", chunks, top_k=5)

        assert result.stage == "fallback:parse"
        assert result.chunks == chunks[:5]
        assert result.usage is not None

    async def test_cancelled_error_propagates(self, enabled, monkeypatch):
        async def cancel(*a, **k):
            raise asyncio.CancelledError()

        monkeypatch.setattr(reranker, "call_with_model", cancel)
        with pytest.raises(asyncio.CancelledError):
            await rerank("q", _chunks(8), top_k=5)

    async def test_prompt_truncates_content(self, monkeypatch):
        monkeypatch.setattr(reranker, "get_settings", lambda: _settings(rerank_max_chunk_chars=100))
        chunk = RetrievedChunk(
            chunk_id=1, source_type="document", source_id="d", title="t",
            content="가" * 500, similarity_score=0.5,
        )
        _, user = reranker.build_rerank_prompt("q", [chunk], 1)
        assert "가" * 100 in user and "가" * 101 not in user

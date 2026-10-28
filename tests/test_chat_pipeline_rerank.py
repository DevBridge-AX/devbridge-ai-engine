"""
chat_pipeline의 리랭크 단계 연결 검증. LLM·검색은 mock으로 대체합니다(네트워크 호출 없음).

- disabled: retrieve top_k=5, 리랭크 LLM 미호출
- enabled: retrieve top_k=rerank_candidates, 컨텍스트/citations가 리랭크 순서, 메트릭 필드 기록
- done payload 스키마 불변(token_usage에 rerank 항목 없음)
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from app.core import chat_pipeline
from app.core.llm.provider import LLMUsage
from app.core.rag import reranker
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest

_CANDIDATES = 8


def _request() -> ChatRequest:
    return ChatRequest(
        session_id="s",
        content="배포 절차 알려줘",
        conversation_history=[],
        workspace_id="ws",
        user_id="u",
        role="planner",
    )


def _all_chunks(n: int) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=i,
            source_type="document",
            source_id=f"doc-{i}",
            title=f"문서{i}",
            content=f"내용{i}",
            similarity_score=round(0.95 - i * 0.05, 2),
        )
        for i in range(1, n + 1)
    ]


def _run(monkeypatch, *, enabled: bool, ranking: list[int] | None = None) -> dict:
    captured: dict = {"metrics": [], "rerank_calls": 0}

    async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None, **kw):
        captured["retrieve_top_k"] = top_k
        return _all_chunks(10)[:top_k]

    async def fake_assess(chunks, query):
        captured["assess_ids"] = [c.chunk_id for c in chunks]
        return GroundingResult(is_groundable=True, confidence=0.9)

    async def fake_call_main_stream(messages, system_prompt, max_tokens=4096):
        captured["system_prompt"] = system_prompt
        yield "답변", None
        yield None, LLMUsage(model="fake-main", prompt_tokens=10, completion_tokens=5)

    async def fake_call_with_model(model, messages, system, **kwargs):
        captured["rerank_calls"] += 1
        return json.dumps({"ranking": ranking}), LLMUsage(model, 77, 6)

    pipeline_settings = SimpleNamespace(
        main_model="fake-main",
        main_max_tokens=256,
        rerank_enabled=enabled,
        rerank_candidates=_CANDIDATES,
    )
    rerank_settings = SimpleNamespace(
        rerank_enabled=enabled,
        rerank_model="gemini-test",
        rerank_candidates=_CANDIDATES,
        rerank_timeout_seconds=2.0,
        rerank_max_chunk_chars=600,
    )
    monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
    monkeypatch.setattr(chat_pipeline.llm, "call_main_stream", fake_call_main_stream)
    monkeypatch.setattr(chat_pipeline, "get_settings", lambda: pipeline_settings)
    monkeypatch.setattr(reranker, "get_settings", lambda: rerank_settings)
    monkeypatch.setattr(reranker, "call_with_model", fake_call_with_model)
    monkeypatch.setattr(
        chat_pipeline, "record_metric", lambda event, payload: captured["metrics"].append((event, payload))
    )

    async def consume():
        return [event async for event in chat_pipeline.run(_request(), db=None)]

    captured["events"] = asyncio.run(consume())
    return captured


def _chat_metric(captured: dict) -> dict:
    return next(p for e, p in captured["metrics"] if e == "chat_metrics")


class TestRerankDisabled:

    def test_disabled_retrieves_five_and_never_calls_llm(self, monkeypatch):
        captured = _run(monkeypatch, enabled=False)

        assert captured["retrieve_top_k"] == 5
        assert captured["rerank_calls"] == 0
        assert captured["assess_ids"] == [1, 2, 3, 4, 5]
        metric = _chat_metric(captured)
        assert metric["rerank_stage"] == "disabled"
        assert metric["rerank_candidates"] == 5
        assert metric["rerank_prompt_tokens"] is None


class TestRerankEnabled:

    @pytest.fixture
    def captured(self, monkeypatch):
        return _run(monkeypatch, enabled=True, ranking=[7, 2, 5, 1, 3])

    def test_retrieve_uses_rerank_candidates(self, captured):
        assert captured["retrieve_top_k"] == _CANDIDATES
        assert captured["rerank_calls"] == 1

    def test_grounding_context_and_citations_follow_rerank_order(self, captured):
        assert captured["assess_ids"] == [7, 2, 5, 1, 3]
        prompt = captured["system_prompt"]
        assert prompt.index("문서7") < prompt.index("문서2") < prompt.index("문서5")
        assert "문서8" not in prompt

        done = captured["events"][-1]
        assert done.event == "done"
        cited = [c["chunk_id"] for c in done.data["citations"]]
        assert cited == [7, 2, 5, 1, 3]
        # similarity_score는 원래 값 유지
        assert done.data["citations"][0]["similarity_score"] == _all_chunks(10)[6].similarity_score

    def test_metric_fields_recorded(self, captured):
        metric = _chat_metric(captured)
        assert metric["rerank_stage"] == "applied"
        assert metric["rerank_prompt_tokens"] == 77
        assert metric["rerank_completion_tokens"] == 6
        assert metric["rerank_candidates"] == _CANDIDATES
        assert metric["rerank_ms"] is not None
        # top_similarity는 리랭크 후 chunks 기준
        assert metric["chunk_count"] == 5

    def test_done_payload_schema_unchanged(self, captured):
        data = captured["events"][-1].data
        assert set(data) == {
            "citations",
            "is_groundable",
            "confidence",
            "suggested_owner_id",
            "prompt_version",
            "token_usage",
        }
        assert set(data["token_usage"]) == {"main", "rewrite", "grounding", "context_truncated"}


class TestRerankFallbackInPipeline:

    def test_parse_failure_keeps_original_order(self, monkeypatch):
        captured = _run(monkeypatch, enabled=True, ranking=None)

        assert captured["assess_ids"] == [1, 2, 3, 4, 5]
        assert _chat_metric(captured)["rerank_stage"] == "fallback:parse"
        assert captured["events"][-1].event == "done"

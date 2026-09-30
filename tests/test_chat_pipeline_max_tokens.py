"""
settings.main_max_tokens가 call_main_stream(max_tokens=...)로 그대로 전달되는지 검증하는
유닛 테스트. LLM·검색은 mock으로 대체합니다(API 호출 없음, non-live).
"""

import asyncio
from types import SimpleNamespace

from app.core import chat_pipeline
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest


def _request(**overrides) -> ChatRequest:
    payload = {
        "session_id": "s",
        "content": "배포 절차 알려줘",
        "conversation_history": [],
        "workspace_id": "ws",
        "user_id": "u",
        "role": "planner",
        **overrides,
    }
    return ChatRequest(**payload)


def _run_pipeline(monkeypatch, main_max_tokens: int) -> dict:
    captured: dict = {}

    async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None):
        return [
            RetrievedChunk(
                chunk_id=1,
                source_type="document",
                source_id="doc-1",
                title="배포 절차",
                content="배포는 blue-green 전략을 사용합니다.",
                similarity_score=0.9,
            )
        ]

    async def fake_assess(chunks, query):
        return GroundingResult(is_groundable=True, confidence=0.9)

    async def fake_call_main_stream(messages, system_prompt, max_tokens=4096):
        captured["max_tokens"] = max_tokens
        yield "안녕", None
        yield None, LLMUsage(model="fake-main", prompt_tokens=10, completion_tokens=5)

    monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
    monkeypatch.setattr(chat_pipeline.llm, "call_main_stream", fake_call_main_stream)
    monkeypatch.setattr(
        chat_pipeline,
        "get_settings",
        lambda: SimpleNamespace(main_model="fake-main", main_max_tokens=main_max_tokens),
    )
    # 이 테스트는 max_tokens 전달만 검증하므로 metrics 기록은 no-op으로 대체(레포에 파일 미생성).
    monkeypatch.setattr(chat_pipeline, "record_metric", lambda event, payload: None)

    async def consume():
        return [event async for event in chat_pipeline.run(_request(), db=None)]

    captured["events"] = asyncio.run(consume())
    return captured


class TestMainMaxTokensForwarded:

    def test_settings_value_forwarded_to_call_main_stream(self, monkeypatch):
        captured = _run_pipeline(monkeypatch, main_max_tokens=256)

        assert captured["max_tokens"] == 256
        assert captured["events"][-1].event == "done"

    def test_different_settings_value_is_also_forwarded(self, monkeypatch):
        """설정값이 바뀌면 호출 인자도 그대로 따라간다(하드코딩 회귀 방지)."""
        captured = _run_pipeline(monkeypatch, main_max_tokens=8000)

        assert captured["max_tokens"] == 8000

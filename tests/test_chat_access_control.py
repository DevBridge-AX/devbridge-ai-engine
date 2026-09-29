"""
/chat 접근 제어 필드 전달 유닛 테스트 (docs/access-control.md §5).

ChatRequest의 accessible_task_ids/can_view_restricted가 하위호환 기본값을 가지며,
chat_pipeline이 이를 AccessFilter로 retriever에 전달하는지 검증합니다.
LLM·검색은 mock으로 대체합니다.
"""

import asyncio
from types import SimpleNamespace

from app.core import chat_pipeline
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import AccessFilter
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


def _run_pipeline(monkeypatch, request: ChatRequest) -> dict:
    captured: dict = {}

    async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None):
        captured["access"] = access
        return []

    async def fake_assess(chunks, query):
        return GroundingResult(is_groundable=False, confidence=0.0)

    monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
    monkeypatch.setattr(
        chat_pipeline, "get_settings", lambda: SimpleNamespace(main_model="fake-main")
    )
    # 이 테스트는 access 필드 전달만 검증하므로 metrics 기록은 no-op으로 대체(레포에 파일 미생성).
    monkeypatch.setattr(chat_pipeline, "record_metric", lambda event, payload: None)

    async def consume():
        return [event async for event in chat_pipeline.run(request, db=None)]

    captured["events"] = asyncio.run(consume())
    return captured


class TestChatRequestAccessFields:

    def test_defaults_are_backward_compatible(self):
        request = _request()
        assert request.accessible_task_ids is None
        assert request.can_view_restricted is False

    def test_fields_accepted(self):
        request = _request(accessible_task_ids=["t1", "t2"], can_view_restricted=True)
        assert request.accessible_task_ids == ["t1", "t2"]
        assert request.can_view_restricted is True


class TestPipelinePassesAccess:

    def test_access_filter_forwarded(self, monkeypatch):
        captured = _run_pipeline(
            monkeypatch, _request(accessible_task_ids=["t1"], can_view_restricted=True)
        )
        assert captured["access"] == AccessFilter(
            accessible_task_ids=["t1"], can_view_restricted=True
        )
        assert captured["events"][-1].event == "done"

    def test_default_request_forwards_default_filter(self, monkeypatch):
        captured = _run_pipeline(monkeypatch, _request())
        assert captured["access"] == AccessFilter()

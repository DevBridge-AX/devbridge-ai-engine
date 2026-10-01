"""
app/pipelines/workspace_summary.py LLM 실패 fallback 유닛 테스트 (API 호출 없음).

AI_ANALYSIS_MODE=llm + GMS_API_KEY 설정 상태에서 call_structured()가 HTTP/JSON
오류를 던지면 generate_workspace_summary()가 이를 잡아 fallback 규칙 요약으로
대체하고 mode="llm_fallback"을 반환하는지, logger.warning이 남는지 확인합니다.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from app.pipelines import workspace_summary
from app.schemas.workspace_summary import WorkspaceSummaryRequest

_DOCUMENT_ANALYSIS_MODEL = "fallback-v1"
_MAIN_MODEL = "claude-test-main"


def _request() -> WorkspaceSummaryRequest:
    return WorkspaceSummaryRequest(
        workspace_id="ws-1",
        workspace_name="DevBridge 개발팀",
        total_task_count=20,
        assigned_task_count=15,
        in_progress_task_count=6,
        done_task_count=10,
        delayed_task_count=1,
        progress_rate=50,
        member_count=5,
        recent_task_count=3,
        recent_git_commit_count=4,
        recent_document_count=2,
    )


@pytest.fixture
def llm_mode_settings(monkeypatch):
    settings = SimpleNamespace(
        ai_analysis_mode="llm",
        gms_api_key="test-key",
        document_analysis_model=_DOCUMENT_ANALYSIS_MODEL,
        main_model=_MAIN_MODEL,
    )
    monkeypatch.setattr(workspace_summary, "get_settings", lambda: settings)
    return settings


def test_llm_http_error_falls_back_with_warning(monkeypatch, caplog, llm_mode_settings):
    async def raise_http_error(**kwargs):
        request = httpx.Request("POST", "http://gms.test/v1/messages")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(workspace_summary, "call_structured", raise_http_error)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(workspace_summary.generate_workspace_summary(_request()))

    assert result.mode == "llm_fallback"
    assert result.model == _DOCUMENT_ANALYSIS_MODEL
    assert result.summary.strip() != ""
    assert any("workspace_summary" in record.message for record in caplog.records)


def test_llm_json_error_falls_back_with_warning(monkeypatch, caplog, llm_mode_settings):
    async def raise_json_error(**kwargs):
        raise json.JSONDecodeError("bad json", "doc", 0)

    monkeypatch.setattr(workspace_summary, "call_structured", raise_json_error)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(workspace_summary.generate_workspace_summary(_request()))

    assert result.mode == "llm_fallback"
    assert any("LLM 요약 실패" in record.message for record in caplog.records)


def test_llm_success_still_returns_llm_mode(monkeypatch, llm_mode_settings):
    async def fake_call_structured(**kwargs):
        return {"summary": "DevBridge 개발팀 워크스페이스는 순조롭게 진행 중입니다."}

    monkeypatch.setattr(workspace_summary, "call_structured", fake_call_structured)

    result = asyncio.run(workspace_summary.generate_workspace_summary(_request()))

    assert result.mode == "llm"
    assert result.model == _MAIN_MODEL
    assert result.summary == "DevBridge 개발팀 워크스페이스는 순조롭게 진행 중입니다."


def test_fallback_mode_untouched(monkeypatch):
    settings = SimpleNamespace(
        ai_analysis_mode="fallback",
        gms_api_key="",
        document_analysis_model=_DOCUMENT_ANALYSIS_MODEL,
        main_model=_MAIN_MODEL,
    )
    monkeypatch.setattr(workspace_summary, "get_settings", lambda: settings)

    result = asyncio.run(workspace_summary.generate_workspace_summary(_request()))

    assert result.mode == "fallback"
    assert result.model == _DOCUMENT_ANALYSIS_MODEL

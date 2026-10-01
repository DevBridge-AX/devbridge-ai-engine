"""
app/pipelines/document_analysis.py LLM 실패 fallback 유닛 테스트 (API 호출 없음).

AI_ANALYSIS_MODE=llm에서 call_structured()가 HTTP/JSON 오류를 던지면
analyze_document()가 이를 잡아 fallback 규칙 분석 결과로 대체하고 mode="llm_fallback"을
반환하는지, logger.warning이 남는지 확인합니다. 정상 llm 성공 경로(mode="llm")는
회귀 방지용으로 함께 검증합니다.
"""

import asyncio
import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from app.pipelines import document_analysis
from app.schemas.analysis import DocumentAnalysisRequest

_DOCUMENT_ANALYSIS_MODEL = "fallback-v1"
_MAIN_MODEL = "claude-test-main"


def _request(tmp_path) -> DocumentAnalysisRequest:
    file_path = tmp_path / "doc.md"
    file_path.write_text("# 배포 절차\n임시 점검 사항이 있습니다.", encoding="utf-8")

    return DocumentAnalysisRequest(
        document_id="doc-1",
        workspace_id="ws-1",
        task_id="task-1",
        title="배포 절차 문서",
        file_path=str(file_path),
        doc_type="spec",
    )


@pytest.fixture
def llm_mode_settings(monkeypatch):
    settings = SimpleNamespace(
        ai_analysis_mode="llm",
        gms_api_key="test-key",
        document_analysis_model=_DOCUMENT_ANALYSIS_MODEL,
        main_model=_MAIN_MODEL,
    )
    monkeypatch.setattr(document_analysis, "get_settings", lambda: settings)
    return settings


def test_llm_http_error_falls_back_with_warning(monkeypatch, tmp_path, caplog, llm_mode_settings):
    async def raise_http_error(**kwargs):
        request = httpx.Request("POST", "http://gms.test/v1/messages")
        response = httpx.Response(500, request=request)
        raise httpx.HTTPStatusError("boom", request=request, response=response)

    monkeypatch.setattr(document_analysis, "call_structured", raise_http_error)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(document_analysis.analyze_document(_request(tmp_path)))

    assert result.mode == "llm_fallback"
    assert result.model == _DOCUMENT_ANALYSIS_MODEL
    assert result.summary.strip() != ""
    assert result.keywords.strip() != ""
    assert result.risk_level in {"LOW", "MEDIUM", "HIGH"}
    assert result.next_action.strip() != ""
    assert any("document_analysis" in record.message for record in caplog.records)


def test_llm_json_error_falls_back_with_warning(monkeypatch, tmp_path, caplog, llm_mode_settings):
    async def raise_json_error(**kwargs):
        raise json.JSONDecodeError("bad json", "doc", 0)

    monkeypatch.setattr(document_analysis, "call_structured", raise_json_error)

    with caplog.at_level(logging.WARNING):
        result = asyncio.run(document_analysis.analyze_document(_request(tmp_path)))

    assert result.mode == "llm_fallback"
    assert result.model == _DOCUMENT_ANALYSIS_MODEL
    assert any("LLM 분석 실패" in record.message for record in caplog.records)


def test_llm_success_still_returns_llm_mode(monkeypatch, tmp_path, llm_mode_settings):
    async def fake_call_structured(**kwargs):
        return {
            "summary": "배포 절차 문서 요약입니다.",
            "keywords": "배포, 절차",
            "risk_level": "LOW",
            "next_action": "다음 배포 일정을 확인하세요.",
        }

    monkeypatch.setattr(document_analysis, "call_structured", fake_call_structured)

    result = asyncio.run(document_analysis.analyze_document(_request(tmp_path)))

    assert result.mode == "llm"
    assert result.model == _MAIN_MODEL
    assert result.summary == "배포 절차 문서 요약입니다."


def test_fallback_mode_untouched(monkeypatch, tmp_path):
    settings = SimpleNamespace(
        ai_analysis_mode="fallback",
        gms_api_key="",
        document_analysis_model=_DOCUMENT_ANALYSIS_MODEL,
        main_model=_MAIN_MODEL,
    )
    monkeypatch.setattr(document_analysis, "get_settings", lambda: settings)

    result = asyncio.run(document_analysis.analyze_document(_request(tmp_path)))

    assert result.mode == "fallback"
    assert result.model == _DOCUMENT_ANALYSIS_MODEL

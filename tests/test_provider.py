"""
app/core/llm/provider.py call_grounding() 유닛 테스트.

httpx.MockTransport로 GMS Gemini 엔드포인트 응답을 흉내내어 외부 호출 없이 검증합니다.
요청 URL/헤더/바디 포맷, 응답 파싱 계약(파싱 실패 시 빈 dict 반환), HTTP 오류 전파를 확인합니다.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.llm import provider
from app.core.llm.provider import LLMUsage, call_grounding

_BASE_URL = "http://gms.test/v1beta"
_MODEL = "gemini-test-lite"
_API_KEY = "test-gms-key"


@pytest.fixture(autouse=True)
def patched_settings(monkeypatch):
    monkeypatch.setattr(
        provider,
        "get_settings",
        lambda: SimpleNamespace(
            grounding_model=_MODEL,
            gemini_base_url=_BASE_URL,
            gms_api_key=_API_KEY,
        ),
    )


@pytest.fixture
def mock_gemini(monkeypatch):
    """handler를 등록하면 provider가 만드는 httpx.AsyncClient가 MockTransport를 사용합니다."""
    captured: list[httpx.Request] = []
    real_client = httpx.AsyncClient

    def install(handler):
        def wrapped(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return handler(request)

        def factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(wrapped)
            return real_client(*args, **kwargs)

        monkeypatch.setattr(provider.httpx, "AsyncClient", factory)
        return captured

    return install


def _gemini_response(text: str, prompt_tokens: int = 120, completion_tokens: int = 15) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": completion_tokens,
        },
    }


class TestCallGroundingNormal:

    async def test_returns_parsed_dict_and_usage(self, mock_gemini):
        mock_gemini(lambda req: httpx.Response(
            200, json=_gemini_response('{"is_groundable": true, "confidence": 0.87}')
        ))

        result, usage = await call_grounding("질문: 배치 스케줄\n\n검색된 컨텍스트:\n...")

        assert result == {"is_groundable": True, "confidence": 0.87}
        assert isinstance(usage, LLMUsage)
        assert usage.model == _MODEL
        assert usage.prompt_tokens == 120
        assert usage.completion_tokens == 15

    async def test_request_url_headers_and_body(self, mock_gemini):
        captured = mock_gemini(lambda req: httpx.Response(
            200, json=_gemini_response('{"is_groundable": false, "confidence": 0.1}')
        ))

        await call_grounding("판정 입력")

        assert len(captured) == 1
        req = captured[0]
        assert req.method == "POST"
        assert str(req.url) == f"{_BASE_URL}/models/{_MODEL}:generateContent"
        assert req.headers["x-goog-api-key"] == _API_KEY
        assert req.headers["content-type"] == "application/json"

        body = json.loads(req.content)
        # system prompt는 user/model 선행 턴으로, 사용자 프롬프트는 마지막 user 턴으로 전달
        assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
        assert body["contents"][0]["parts"][0]["text"] == provider._GROUNDING_SYSTEM_PROMPT
        assert body["contents"][-1] == {"role": "user", "parts": [{"text": "판정 입력"}]}

        gen_config = body["generationConfig"]
        assert gen_config["maxOutputTokens"] == 64
        assert gen_config["responseMimeType"] == "application/json"
        schema = gen_config["responseSchema"]
        assert schema["type"] == "OBJECT"
        assert set(schema["required"]) == {"is_groundable", "confidence"}
        assert schema["properties"]["is_groundable"] == {"type": "BOOLEAN"}
        assert schema["properties"]["confidence"] == {"type": "NUMBER"}

    async def test_strips_markdown_code_fence(self, mock_gemini):
        fenced = '```json\n{"is_groundable": true, "confidence": 0.5}\n```'
        mock_gemini(lambda req: httpx.Response(200, json=_gemini_response(fenced)))

        result, _ = await call_grounding("p")

        assert result == {"is_groundable": True, "confidence": 0.5}


class TestCallGroundingMalformed:
    """grounding.assess()는 빈 dict/누락 키를 파싱 실패로 간주해 유사도 fallback을 적용한다."""

    async def test_non_json_text_returns_empty_dict_with_usage(self, mock_gemini):
        mock_gemini(lambda req: httpx.Response(
            200, json=_gemini_response("답변 가능합니다.", prompt_tokens=10, completion_tokens=3)
        ))

        result, usage = await call_grounding("p")

        assert result == {}
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 3

    async def test_missing_fields_returned_as_is(self, mock_gemini):
        mock_gemini(lambda req: httpx.Response(
            200, json=_gemini_response('{"is_groundable": true}')
        ))

        result, _ = await call_grounding("p")

        assert result == {"is_groundable": True}
        assert result.get("confidence") is None

    async def test_no_candidates_returns_empty_dict_and_zero_usage(self, mock_gemini):
        mock_gemini(lambda req: httpx.Response(200, json={}))

        result, usage = await call_grounding("p")

        assert result == {}
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0


class TestCallGroundingThinkingWarning:

    async def test_call_grounding_warns_on_max_tokens_finish(self, mock_gemini, caplog):
        """finishReason=MAX_TOKENS(thinking 모델 truncate) 시 경고 로그가 남아야 합니다."""
        response = {
            "candidates": [
                {"content": {"parts": []}, "finishReason": "MAX_TOKENS"},
            ],
            "usageMetadata": {
                "promptTokenCount": 120,
                "candidatesTokenCount": 0,
                "thoughtsTokenCount": 64,
            },
        }
        mock_gemini(lambda req: httpx.Response(200, json=response))

        with caplog.at_level("WARNING", logger="app.core.llm.provider"):
            result, usage = await call_grounding("p")

        assert result == {}
        assert usage.prompt_tokens == 120
        assert any(
            "thinking/truncation" in record.message
            and f"model={_MODEL}" in record.message
            and "finish=MAX_TOKENS" in record.message
            and "thoughts=64" in record.message
            for record in caplog.records
        )


class TestCallGroundingHttpError:

    @pytest.mark.parametrize("status", [400, 401, 429, 500, 503])
    async def test_error_status_propagates(self, mock_gemini, status):
        mock_gemini(lambda req: httpx.Response(status, json={"error": "boom"}))

        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await call_grounding("p")

        assert exc_info.value.response.status_code == status

    async def test_transport_error_propagates(self, mock_gemini):
        def handler(req):
            raise httpx.ConnectError("connection refused")

        mock_gemini(handler)

        with pytest.raises(httpx.ConnectError):
            await call_grounding("p")

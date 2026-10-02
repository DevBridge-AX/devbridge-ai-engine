"""
app/core/llm/provider.py 호출 단위 관측성(llm_calls 이벤트) 유닛 테스트.

httpx.MockTransport로 GMS 엔드포인트 응답을 흉내내어 외부 호출 없이 검증합니다.
chat_metrics(요청 단위)와 달리 LLM 호출 1건마다 llm_calls.jsonl에 정확히 1줄이
기록되는지, 토큰/finish_reason/thoughts_tokens 계약과 프롬프트/응답 원문 미포함을
검증합니다. metrics_dir는 tmp_path로 돌립니다.
"""

import json
from types import SimpleNamespace

import httpx
import pytest

from app.core import metrics
from app.core.llm import provider
from app.core.llm.provider import call_grounding, call_main, call_main_stream

_ANTHROPIC_BASE_URL = "http://gms.test/anthropic"
_OPENAI_BASE_URL = "http://gms.test/openai"
_GEMINI_BASE_URL = "http://gms.test/v1beta"
_API_KEY = "test-gms-key"
_MAIN_MODEL = "claude-test-main"
_GROUNDING_MODEL = "gemini-test-lite"


@pytest.fixture(autouse=True)
def patched_settings(monkeypatch):
    monkeypatch.setattr(
        provider,
        "get_settings",
        lambda: SimpleNamespace(
            main_model=_MAIN_MODEL,
            rewrite_model=_MAIN_MODEL,
            grounding_model=_GROUNDING_MODEL,
            anthropic_base_url=_ANTHROPIC_BASE_URL,
            openai_base_url=_OPENAI_BASE_URL,
            gemini_base_url=_GEMINI_BASE_URL,
            gms_api_key=_API_KEY,
        ),
    )


@pytest.fixture(autouse=True)
def metrics_tmp_dir(monkeypatch, tmp_path):
    """record_metric이 실제로 사용하는 app.core.metrics.get_settings를 tmp_path로 돌립니다."""
    monkeypatch.setattr(
        metrics,
        "get_settings",
        lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(tmp_path)),
    )
    return tmp_path


@pytest.fixture
def mock_transport(monkeypatch):
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


def _read_records(tmp_path, event: str) -> list[dict]:
    path = tmp_path / f"{event}.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _anthropic_response(
    text: str, input_tokens: int = 50, output_tokens: int = 10, stop_reason: str = "end_turn"
) -> dict:
    return {
        "content": [{"type": "text", "text": text}],
        "stop_reason": stop_reason,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def _gemini_response(
    text: str,
    prompt_tokens: int = 120,
    candidates_tokens: int = 15,
    thoughts_tokens: int = 0,
    finish_reason: str = "STOP",
) -> dict:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish_reason}],
        "usageMetadata": {
            "promptTokenCount": prompt_tokens,
            "candidatesTokenCount": candidates_tokens,
            "thoughtsTokenCount": thoughts_tokens,
        },
    }


class TestCallMainRecordsLLMCall:

    async def test_call_main_records_llm_call(self, mock_transport, metrics_tmp_dir):
        mock_transport(lambda req: httpx.Response(200, json=_anthropic_response("안녕하세요")))

        text, usage = await call_main([{"role": "user", "content": "질문"}], "system")

        assert text == "안녕하세요"
        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        record = records[0]
        assert record["purpose"] == "main"
        assert record["model"] == _MAIN_MODEL
        assert record["provider"] == "anthropic"
        assert record["streamed"] is False
        assert record["prompt_tokens"] == usage.prompt_tokens == 50
        assert record["completion_tokens"] == usage.completion_tokens == 10
        assert record["thoughts_tokens"] == 0
        assert record["finish_reason"] == "end_turn"
        assert record["parse_ok"] is True
        assert record["error_type"] is None
        assert record["http_status"] is None
        assert record["latency_ms"] >= 0


class TestCallMainStreamRecordsOnce:

    async def test_stream_records_once_with_ttft(self, mock_transport, metrics_tmp_dir):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 42}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "안녕"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "하세요"}},
            {"type": "message_delta", "usage": {"output_tokens": 7}},
        ]
        sse_body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)

        mock_transport(
            lambda req: httpx.Response(
                200, content=sse_body.encode("utf-8"), headers={"content-type": "text/event-stream"}
            )
        )

        chunks = []
        final_usage = None
        async for text, usage in call_main_stream([{"role": "user", "content": "질문"}], "system"):
            if text is not None:
                chunks.append(text)
            if usage is not None:
                final_usage = usage

        assert "".join(chunks) == "안녕하세요"
        assert final_usage.prompt_tokens == 42
        assert final_usage.completion_tokens == 7

        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        record = records[0]
        assert record["purpose"] == "main_stream"
        assert record["streamed"] is True
        assert record["ttft_ms"] is not None
        assert record["ttft_ms"] >= 0
        assert record["prompt_tokens"] == 42
        assert record["completion_tokens"] == 7

    async def test_stream_records_finish_reason(self, mock_transport, metrics_tmp_dir):
        events = [
            {"type": "message_start", "message": {"usage": {"input_tokens": 42}}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "안녕"}},
            {"type": "content_block_delta", "delta": {"type": "text_delta", "text": "하세요"}},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 7}},
        ]
        sse_body = "".join(f"data: {json.dumps(e)}\n\n" for e in events)

        mock_transport(
            lambda req: httpx.Response(
                200, content=sse_body.encode("utf-8"), headers={"content-type": "text/event-stream"}
            )
        )

        async for _ in call_main_stream([{"role": "user", "content": "질문"}], "system"):
            pass

        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        assert records[0]["finish_reason"] == "end_turn"


class TestCallGroundingParseFailure:

    async def test_grounding_parse_failure_records_parse_ok_false(self, mock_transport, metrics_tmp_dir):
        mock_transport(
            lambda req: httpx.Response(200, json=_gemini_response("이것은 JSON이 아닙니다."))
        )

        result, usage = await call_grounding("판정 입력")

        assert result == {}
        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        record = records[0]
        assert record["purpose"] == "grounding"
        assert record["parse_ok"] is False
        assert record["error_type"] is None


class TestHttpErrorRecordsAndReraises:

    async def test_http_error_records_error_type_and_reraises(self, mock_transport, metrics_tmp_dir):
        mock_transport(lambda req: httpx.Response(500, json={"error": "boom"}))

        with pytest.raises(httpx.HTTPStatusError):
            await call_main([{"role": "user", "content": "질문"}], "system")

        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        record = records[0]
        assert record["purpose"] == "main"
        assert record["error_type"] == "HTTPStatusError"
        assert record["http_status"] == 500


class TestGeminiThoughtsTokens:

    async def test_gemini_thoughts_tokens_added_to_completion(self, mock_transport, metrics_tmp_dir):
        mock_transport(
            lambda req: httpx.Response(
                200,
                json=_gemini_response(
                    '{"is_groundable": true, "confidence": 0.9}',
                    prompt_tokens=120,
                    candidates_tokens=5,
                    thoughts_tokens=64,
                    finish_reason="STOP",
                ),
            )
        )

        result, usage = await call_grounding("판정 입력")

        assert result == {"is_groundable": True, "confidence": 0.9}
        # 과금 기준: completion_tokens == candidatesTokenCount + thoughtsTokenCount
        assert usage.completion_tokens == 5 + 64

        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        record = records[0]
        assert record["completion_tokens"] == 69
        assert record["thoughts_tokens"] == 64


class TestPayloadHasNoPromptText:

    async def test_payload_has_no_prompt_text(self, mock_transport, metrics_tmp_dir):
        secret_system = "SECRET_SYSTEM_PROMPT_XYZ"
        secret_user = "SECRET_USER_MESSAGE_ABC"
        secret_response = "SECRET_RESPONSE_TEXT_123"

        mock_transport(lambda req: httpx.Response(200, json=_anthropic_response(secret_response)))

        await call_main([{"role": "user", "content": secret_user}], secret_system)

        path = metrics_tmp_dir / "llm_calls.jsonl"
        raw = path.read_text(encoding="utf-8")

        assert secret_system not in raw
        assert secret_user not in raw
        assert secret_response not in raw


class TestCancelledErrorRecording:
    """wait_for 타임아웃 등으로 취소된 호출이 성공(error_type=None)으로 기록되지 않아야 합니다."""

    async def test_call_rewrite_cancelled_records_error_type(self, monkeypatch, metrics_tmp_dir):
        import asyncio

        async def slow_request(url, headers, body, timeout):
            await asyncio.sleep(10)

        monkeypatch.setattr(provider, "_request_once", slow_request)

        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(
                provider.call_rewrite(
                    [{"role": "user", "content": "q"}], "sys", max_tokens=8, purpose="cache_verify"
                ),
                timeout=0.05,
            )

        records = _read_records(metrics_tmp_dir, "llm_calls")
        assert len(records) == 1
        assert records[0]["purpose"] == "cache_verify"
        assert records[0]["error_type"] == "CancelledError"

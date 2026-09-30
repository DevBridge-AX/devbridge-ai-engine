"""
LLM API 클라이언트 — GMS 멀티 프로바이더 게이트웨이.

모든 LLM 호출은 이 모듈의 함수를 통해서만 이루어집니다.
모델명 prefix로 provider를 자동 감지하며, 단일 GMS_API_KEY로 인증합니다.
모델 교체는 .env 값만 변경하면 됩니다 (코드 변경 불필요).

call_main()        — 메인 답변 생성 (non-streaming)              MAIN_MODEL    (claude-*)
call_main_stream() — 메인 답변 생성 (Anthropic SSE 스트리밍)     MAIN_MODEL    (claude-* 전용, SSE 포맷이 Anthropic 고정)
call_rewrite()     — 멀티턴 쿼리 재구성                           REWRITE_MODEL (provider-prefix 무관, claude-*/gpt-*/gemini-* 모두 가능)
call_grounding()   — 그라운딩 이진 판정, JSON dict 반환           GROUNDING_MODEL (gemini-*)
call_structured()  — JSON 구조화 응답 범용                        MAIN_MODEL    (claude-*)

provider 자동 감지:
  claude-* → anthropic  (URL: anthropic_base_url/v1/messages, 헤더: x-api-key)
  gpt-*/o* → openai     (URL: openai_base_url/chat/completions, 헤더: Authorization Bearer)
  gemini-* → gemini     (URL: gemini_base_url/models/{model}:generateContent, 헤더: x-goog-api-key)

스트리밍(SSE)은 call_main_stream()의 Anthropic 포맷만 지원합니다.
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.core.metrics import record_metric
from app.core.rag.grounding_prompts import GROUNDING_PROMPT_V1, get_grounding_prompt

_ANTHROPIC_VERSION = "2023-06-01"
logger = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """LLM 호출 토큰 사용량. chat 엔드포인트에서 TokenUsageDetail로 변환됩니다."""

    model: str
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------

def _record_llm_call(
    *,
    purpose: str,
    model: str,
    provider: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    thoughts_tokens: int,
    finish_reason: str | None,
    parse_ok: bool,
    error_type: str | None,
    http_status: int | None,
    streamed: bool,
    ttft_ms: float | None = None,
) -> None:
    """LLM 호출 1건의 관측성 이벤트를 `llm_calls.jsonl`에 기록합니다.

    chat_metrics(요청 단위)와 별도로, 호출 1건 단위로 튜닝용 세부 지표를 남깁니다.
    payload에는 프롬프트/메시지/시스템 프롬프트/응답 원문을 절대 포함하지 않습니다.
    """
    record_metric(
        "llm_calls",
        {
            "purpose": purpose,
            "model": model,
            "provider": provider,
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts_tokens,
            "finish_reason": finish_reason,
            "parse_ok": parse_ok,
            "error_type": error_type,
            "http_status": http_status,
            "streamed": streamed,
            "ttft_ms": ttft_ms,
        },
    )


def _detect_provider(model: str) -> str:
    """모델명 prefix로 GMS provider를 자동 감지합니다."""
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gpt") or model.startswith("o"):
        return "openai"
    if model.startswith("gemini"):
        return "gemini"
    raise ValueError(f"Unknown model prefix: {model!r}")


def _build_url(provider: str, model: str, settings) -> str:
    if provider == "anthropic":
        return f"{settings.anthropic_base_url}/v1/messages"
    if provider == "openai":
        return f"{settings.openai_base_url}/chat/completions"
    # gemini
    return f"{settings.gemini_base_url}/models/{model}:generateContent"


def _build_headers(provider: str, settings) -> dict:
    if provider == "anthropic":
        return {
            "x-api-key": settings.gms_api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
    if provider == "openai":
        return {
            "Authorization": f"Bearer {settings.gms_api_key}",
            "content-type": "application/json",
        }
    # gemini
    return {
        "x-goog-api-key": settings.gms_api_key,
        "content-type": "application/json",
    }


def _build_body(
    provider: str,
    model: str,
    messages: list[dict],
    system: str = "",
    max_tokens: int = 4096,
    stream: bool = False,
    json_output: bool = False,
    response_schema: dict | None = None,
) -> dict:
    """Provider별 요청 바디를 구성합니다.

    messages는 OpenAI 형식(role: user/assistant, content: str)으로 전달하며,
    각 provider 포맷으로 변환됩니다.

    system 처리 방식:
      anthropic — "system" 최상위 필드
      openai    — {"role": "developer", "content": system} messages 맨 앞 삽입
      gemini    — user/model 선행 턴으로 contents 앞에 삽입
    """
    if provider == "anthropic":
        body: dict = {"model": model, "max_tokens": max_tokens, "messages": messages}
        if system:
            body["system"] = system
        if stream:
            body["stream"] = True
        return body

    if provider == "openai":
        oai_messages: list[dict] = []
        if system:
            oai_messages.append({"role": "developer", "content": system})
        oai_messages.extend(messages)
        body = {"model": model, "messages": oai_messages}
        if json_output:
            body["response_format"] = {"type": "json_object"}
        return body

    # gemini
    contents: list[dict] = []
    if system:
        contents.append({"role": "user", "parts": [{"text": system}]})
        contents.append({"role": "model", "parts": [{"text": "알겠습니다."}]})
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    gen_config: dict = {"maxOutputTokens": max_tokens}
    if json_output:
        gen_config["responseMimeType"] = "application/json"
    if response_schema:
        gen_config["responseMimeType"] = "application/json"
        gen_config["responseSchema"] = response_schema
    return {"contents": contents, "generationConfig": gen_config}


def _parse_response(provider: str, data: dict) -> tuple[str, dict]:
    """응답 JSON에서 텍스트와 토큰 사용량을 추출합니다.

    Returns:
        (text, {"input_tokens": int, "output_tokens": int,
                "thoughts_tokens": int, "finish_reason": str | None})
        usage 키는 provider에 관계없이 통일된 형식으로 반환됩니다.
        thoughts_tokens는 gemini thinking 모델의 추론 토큰(usageMetadata.thoughtsTokenCount)이며,
        gemini의 output_tokens(=completion_tokens)에는 과금 기준에 맞춰 이미 합산되어 있습니다
        (candidatesTokenCount + thoughtsTokenCount). anthropic/openai는 항상 0입니다.
        finish_reason은 gemini candidates[0].finishReason / anthropic stop_reason /
        openai choices[0].finish_reason입니다.
    """
    if provider == "anthropic":
        return (
            data["content"][0]["text"],
            {
                "input_tokens": data["usage"]["input_tokens"],
                "output_tokens": data["usage"]["output_tokens"],
                "thoughts_tokens": 0,
                "finish_reason": data.get("stop_reason"),
            },
        )
    if provider == "openai":
        return (
            data["choices"][0]["message"]["content"],
            {
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"],
                "thoughts_tokens": 0,
                "finish_reason": data["choices"][0].get("finish_reason"),
            },
        )
    # gemini
    candidates = data.get("candidates", [])
    text = ""
    if candidates and "content" in candidates[0]:
        parts = candidates[0]["content"].get("parts", [])
        if parts:
            text = parts[0].get("text", "")

    usage_metadata = data.get("usageMetadata", {})
    candidates_tokens = usage_metadata.get("candidatesTokenCount", 0)
    thoughts_tokens = usage_metadata.get("thoughtsTokenCount", 0)
    finish_reason = candidates[0].get("finishReason") if candidates else None
    return text, {
        "input_tokens": usage_metadata.get("promptTokenCount", 0),
        # 과금 기준(candidatesTokenCount + thoughtsTokenCount)에 맞춘 completion_tokens.
        "output_tokens": candidates_tokens + thoughts_tokens,
        "thoughts_tokens": thoughts_tokens,
        "finish_reason": finish_reason,
    }


async def _request_once(url: str, headers: dict, body: dict, timeout: float) -> dict:
    """POST 요청 1회를 수행하고 응답 JSON을 반환합니다. (기록/측정은 호출부 책임)"""
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, headers=headers, json=body)
        resp.raise_for_status()
        return resp.json()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def call_main(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 4096,
) -> tuple[str, LLMUsage]:
    """MAIN_MODEL(claude-*)로 단일 응답을 생성합니다."""
    settings = get_settings()
    provider = _detect_provider(settings.main_model)
    body = _build_body(provider, settings.main_model, messages, system_prompt, max_tokens)

    start = time.perf_counter()
    prompt_tokens = completion_tokens = thoughts_tokens = 0
    finish_reason: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    try:
        data = await _request_once(
            _build_url(provider, settings.main_model, settings),
            _build_headers(provider, settings),
            body,
            120.0,
        )
        text, usage_raw = _parse_response(provider, data)
        prompt_tokens = usage_raw["input_tokens"]
        completion_tokens = usage_raw["output_tokens"]
        thoughts_tokens = usage_raw["thoughts_tokens"]
        finish_reason = usage_raw["finish_reason"]
        return text, LLMUsage(
            model=settings.main_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            http_status = exc.response.status_code
        raise
    finally:
        _record_llm_call(
            purpose="main",
            model=settings.main_model,
            provider=provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            finish_reason=finish_reason,
            parse_ok=True,
            error_type=error_type,
            http_status=http_status,
            streamed=False,
        )


async def call_main_stream(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 4096,
) -> AsyncGenerator[tuple[str | None, LLMUsage | None], None]:
    """MAIN_MODEL Anthropic SSE 스트리밍. (text_chunk, None) 반복 후 (None, LLMUsage) 1회 종료."""
    settings = get_settings()
    provider = _detect_provider(settings.main_model)
    url = _build_url(provider, settings.main_model, settings)
    headers = _build_headers(provider, settings)
    body = _build_body(provider, settings.main_model, messages, system_prompt, max_tokens, stream=True)

    input_tokens = 0
    output_tokens = 0
    finish_reason: str | None = None
    start = time.perf_counter()
    ttft_ms: float | None = None
    error_type: str | None = None
    http_status: int | None = None

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream("POST", url, headers=headers, json=body) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if not payload:
                        continue
                    try:
                        data = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    event_type = data.get("type")
                    if event_type == "message_start":
                        input_tokens = data["message"]["usage"].get("input_tokens", 0)
                    elif event_type == "content_block_delta":
                        delta = data.get("delta", {})
                        if delta.get("type") == "text_delta":
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start) * 1000
                            yield delta.get("text", ""), None
                    elif event_type == "message_delta":
                        output_tokens = data.get("usage", {}).get("output_tokens", 0)
                        finish_reason = data.get("delta", {}).get("stop_reason")

        yield None, LLMUsage(
            model=settings.main_model,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            http_status = exc.response.status_code
        raise
    finally:
        _record_llm_call(
            purpose="main_stream",
            model=settings.main_model,
            provider=provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=input_tokens,
            completion_tokens=output_tokens,
            thoughts_tokens=0,
            finish_reason=finish_reason,
            parse_ok=True,
            error_type=error_type,
            http_status=http_status,
            streamed=True,
            ttft_ms=ttft_ms,
        )


async def call_rewrite(
    messages: list[dict],
    system_prompt: str = "",
    max_tokens: int = 512,
) -> tuple[str, LLMUsage]:
    """REWRITE_MODEL로 멀티턴 쿼리를 재구성합니다. query_rewriter.py에서 호출됩니다.

    provider-prefix에 무관하게 동작합니다 (claude-*/gpt-*/gemini-* 모두 가능).
    """
    settings = get_settings()
    provider = _detect_provider(settings.rewrite_model)
    body = _build_body(provider, settings.rewrite_model, messages, system_prompt, max_tokens)

    start = time.perf_counter()
    prompt_tokens = completion_tokens = thoughts_tokens = 0
    finish_reason: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    try:
        data = await _request_once(
            _build_url(provider, settings.rewrite_model, settings),
            _build_headers(provider, settings),
            body,
            30.0,
        )
        text, usage_raw = _parse_response(provider, data)
        prompt_tokens = usage_raw["input_tokens"]
        completion_tokens = usage_raw["output_tokens"]
        thoughts_tokens = usage_raw["thoughts_tokens"]
        finish_reason = usage_raw["finish_reason"]
        return text, LLMUsage(
            model=settings.rewrite_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            http_status = exc.response.status_code
        raise
    finally:
        _record_llm_call(
            purpose="rewrite",
            model=settings.rewrite_model,
            provider=provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            finish_reason=finish_reason,
            parse_ok=True,
            error_type=error_type,
            http_status=http_status,
            streamed=False,
        )


_GROUNDING_SYSTEM_PROMPT = GROUNDING_PROMPT_V1


def _default_grounding_prompt(settings) -> str:
    return get_grounding_prompt(getattr(settings, "grounding_prompt_version", "v1"))


_GROUNDING_MAX_TOKENS = 64
_GROUNDING_THOUGHTS_WARN_RATIO = 0.5


async def call_grounding(
    prompt: str, *, system_prompt: str | None = None
) -> tuple[dict, LLMUsage]:
    """GROUNDING_MODEL(gemini-*)로 그라운딩 이진 판정을 수행하고 파싱된 dict를 반환합니다.

    Args:
        prompt: "질문: ...\n\n검색된 컨텍스트:\n..." 형식의 판정 입력.

    Returns:
        ({"is_groundable": bool, "confidence": float}, LLMUsage).
        JSON 파싱 실패(비-JSON 응답, candidates 누락 등) 시 ({}, usage)를 반환합니다.
        thinking 모델(gemini-2.5-flash, gemini-3.5-flash 등)은 max_tokens=64에서
        추론 토큰만 소비하고 truncate되어 빈 응답이 반환되는 경우가 있으며, 이때도
        동일하게 ({}, usage)가 반환됩니다. finishReason이 MAX_TOKENS이거나 thoughts 토큰이
        max_tokens의 50% 이상일 때만 경고 로그를 남깁니다(flash-lite는 매 호출 1~2
        thoughts 토큰을 보고하므로 그 이하는 경고하지 않으며, 지표 기록은 항상 수행).
    """
    settings = get_settings()
    provider = _detect_provider(settings.grounding_model)
    messages = [{"role": "user", "content": prompt}]
    grounding_schema = {
        "type": "OBJECT",
        "properties": {
            "is_groundable": {"type": "BOOLEAN"},
            "confidence": {"type": "NUMBER"},
        },
        "required": ["is_groundable", "confidence"],
    }
    body = _build_body(
        provider,
        settings.grounding_model,
        messages,
        system_prompt if system_prompt is not None else _default_grounding_prompt(settings),
        max_tokens=_GROUNDING_MAX_TOKENS,
        response_schema=grounding_schema,
    )

    start = time.perf_counter()
    prompt_tokens = completion_tokens = thoughts_tokens = 0
    finish_reason: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    parse_ok = True
    try:
        data = await _request_once(
            _build_url(provider, settings.grounding_model, settings),
            _build_headers(provider, settings),
            body,
            30.0,
        )
        text, usage_raw = _parse_response(provider, data)
        prompt_tokens = usage_raw["input_tokens"]
        completion_tokens = usage_raw["output_tokens"]
        thoughts_tokens = usage_raw["thoughts_tokens"]
        finish_reason = usage_raw["finish_reason"]
        if (
            finish_reason == "MAX_TOKENS"
            or thoughts_tokens
            >= _GROUNDING_MAX_TOKENS * _GROUNDING_THOUGHTS_WARN_RATIO
        ):
            logger.warning(
                "call_grounding: thinking/truncation 감지 model=%s finish=%s thoughts=%d",
                settings.grounding_model,
                finish_reason,
                thoughts_tokens,
            )

        text = text.strip()
        usage = LLMUsage(
            model=settings.grounding_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(text), usage
        except json.JSONDecodeError:
            logger.warning("call_grounding: JSON 파싱 실패. 응답: %r", text)
            parse_ok = False
            return {}, usage
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            http_status = exc.response.status_code
        raise
    finally:
        _record_llm_call(
            purpose="grounding",
            model=settings.grounding_model,
            provider=provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            finish_reason=finish_reason,
            parse_ok=parse_ok,
            error_type=error_type,
            http_status=http_status,
            streamed=False,
        )


async def call_structured(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 1024,
    purpose: str = "structured",
) -> dict:
    """MAIN_MODEL로 JSON 구조화 응답을 생성합니다.

    system_prompt에서 반드시 JSON 형식 응답을 명시해야 합니다.
    마크다운 코드 블록(```json...```)이 포함된 경우 자동으로 제거합니다.

    purpose: llm_calls 이벤트 분류용(기본 "structured"). 호출부가 자신의 용도
    (예: "document_analysis", "commit_analysis", "workspace_summary")를 전달합니다.
    반환값/시그니처는 변경하지 않으며, 토큰 사용량은 반환하지 않고 llm_calls
    이벤트로만 노출합니다.
    """
    settings = get_settings()
    provider = _detect_provider(settings.main_model)
    body = _build_body(provider, settings.main_model, messages, system_prompt, max_tokens)

    start = time.perf_counter()
    prompt_tokens = completion_tokens = thoughts_tokens = 0
    finish_reason: str | None = None
    error_type: str | None = None
    http_status: int | None = None
    parse_ok = True
    try:
        data = await _request_once(
            _build_url(provider, settings.main_model, settings),
            _build_headers(provider, settings),
            body,
            120.0,
        )
        text, usage_raw = _parse_response(provider, data)
        prompt_tokens = usage_raw["input_tokens"]
        completion_tokens = usage_raw["output_tokens"]
        thoughts_tokens = usage_raw["thoughts_tokens"]
        finish_reason = usage_raw["finish_reason"]

        text = text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            parse_ok = False
            raise
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, httpx.HTTPStatusError):
            http_status = exc.response.status_code
        raise
    finally:
        _record_llm_call(
            purpose=purpose,
            model=settings.main_model,
            provider=provider,
            latency_ms=(time.perf_counter() - start) * 1000,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thoughts_tokens=thoughts_tokens,
            finish_reason=finish_reason,
            parse_ok=parse_ok,
            error_type=error_type,
            http_status=http_status,
            streamed=False,
        )

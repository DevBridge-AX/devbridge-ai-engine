"""
LLM API 클라이언트 (Anthropic Claude).

모든 LLM 호출은 이 모듈의 함수를 통해서만 이루어집니다.
모델 교체는 config.py의 MAIN_MODEL / REWRITE_MODEL 값만 변경하면 됩니다.

call_main()        — 메인 답변 생성 (non-streaming)
call_main_stream() — 메인 답변 생성 (SSE 스트리밍 async generator)
call_rewrite()     — 쿼리 재구성 (경량 모델)
call_structured()  — JSON 구조화 응답 (grounding 판정 등)
"""

import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx

from app.config import get_settings

_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
_ANTHROPIC_VERSION = "2023-06-01"


@dataclass
class LLMUsage:
    """LLM 호출 토큰 사용량. chat 엔드포인트에서 TokenUsageDetail로 변환됩니다."""

    model: str
    prompt_tokens: int
    completion_tokens: int


def _headers(api_key: str) -> dict:
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


async def call_main(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 4096,
) -> tuple[str, LLMUsage]:
    """MAIN_MODEL으로 단일 응답을 생성합니다."""
    settings = get_settings()
    body = {
        "model": settings.main_model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
    }
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(_ANTHROPIC_API_URL, headers=_headers(settings.anthropic_api_key), json=body)
        resp.raise_for_status()
        data = resp.json()

    text = data["content"][0]["text"]
    usage = LLMUsage(
        model=settings.main_model,
        prompt_tokens=data["usage"]["input_tokens"],
        completion_tokens=data["usage"]["output_tokens"],
    )
    return text, usage


async def call_main_stream(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 4096,
) -> AsyncGenerator[tuple[str | None, LLMUsage | None], None]:
    """MAIN_MODEL 스트리밍 응답. (text_chunk, None) 반복 후 (None, LLMUsage) 1회 종료."""
    settings = get_settings()
    body = {
        "model": settings.main_model,
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": messages,
        "stream": True,
    }
    input_tokens = 0
    output_tokens = 0

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            _ANTHROPIC_API_URL,
            headers=_headers(settings.anthropic_api_key),
            json=body,
        ) as response:
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
                        yield delta.get("text", ""), None
                elif event_type == "message_delta":
                    output_tokens = data.get("usage", {}).get("output_tokens", 0)

    yield None, LLMUsage(
        model=settings.main_model,
        prompt_tokens=input_tokens,
        completion_tokens=output_tokens,
    )


async def call_rewrite(
    messages: list[dict],
    system_prompt: str = "",
    max_tokens: int = 512,
) -> tuple[str, LLMUsage]:
    """REWRITE_MODEL으로 경량 추론을 수행합니다. query_rewriter.py에서 호출됩니다."""
    settings = get_settings()
    body: dict = {
        "model": settings.rewrite_model,
        "max_tokens": max_tokens,
        "messages": messages,
    }
    if system_prompt:
        body["system"] = system_prompt

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(_ANTHROPIC_API_URL, headers=_headers(settings.anthropic_api_key), json=body)
        resp.raise_for_status()
        data = resp.json()

    text = data["content"][0]["text"]
    usage = LLMUsage(
        model=settings.rewrite_model,
        prompt_tokens=data["usage"]["input_tokens"],
        completion_tokens=data["usage"]["output_tokens"],
    )
    return text, usage


async def call_structured(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 1024,
) -> dict:
    """MAIN_MODEL으로 JSON 구조화 응답을 생성합니다.

    system_prompt에서 반드시 JSON 형식 응답을 명시해야 합니다.
    마크다운 코드 블록(```json...```)이 포함된 경우 자동으로 제거합니다.
    """
    text, _ = await call_main(messages, system_prompt, max_tokens=max_tokens)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)

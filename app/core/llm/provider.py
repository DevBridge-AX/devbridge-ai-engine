"""
LLM API 클라이언트 — GMS 멀티 프로바이더 게이트웨이.

모든 LLM 호출은 이 모듈의 함수를 통해서만 이루어집니다.
모델명 prefix로 provider를 자동 감지하며, 단일 GMS_API_KEY로 인증합니다.
모델 교체는 .env 값만 변경하면 됩니다 (코드 변경 불필요).

call_main()        — 메인 답변 생성 (non-streaming)              MAIN_MODEL    (claude-*)
call_main_stream() — 메인 답변 생성 (Anthropic SSE 스트리밍)     MAIN_MODEL    (claude-*)
call_rewrite()     — 멀티턴 쿼리 재구성                           REWRITE_MODEL (gpt-*)
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
from collections.abc import AsyncGenerator
from dataclasses import dataclass

import httpx

from app.config import get_settings

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
        (text, {"input_tokens": int, "output_tokens": int})
        usage 키는 provider에 관계없이 통일된 형식으로 반환됩니다.
    """
    if provider == "anthropic":
        return (
            data["content"][0]["text"],
            {
                "input_tokens": data["usage"]["input_tokens"],
                "output_tokens": data["usage"]["output_tokens"],
            },
        )
    if provider == "openai":
        return (
            data["choices"][0]["message"]["content"],
            {
                "input_tokens": data["usage"]["prompt_tokens"],
                "output_tokens": data["usage"]["completion_tokens"],
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
    return text, {
        "input_tokens": usage_metadata.get("promptTokenCount", 0),
        "output_tokens": usage_metadata.get("candidatesTokenCount", 0),
    }


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

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            _build_url(provider, settings.main_model, settings),
            headers=_build_headers(provider, settings),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    text, usage_raw = _parse_response(provider, data)
    return text, LLMUsage(
        model=settings.main_model,
        prompt_tokens=usage_raw["input_tokens"],
        completion_tokens=usage_raw["output_tokens"],
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
    """REWRITE_MODEL(gpt-*)로 멀티턴 쿼리를 재구성합니다. query_rewriter.py에서 호출됩니다."""
    settings = get_settings()
    provider = _detect_provider(settings.rewrite_model)
    body = _build_body(provider, settings.rewrite_model, messages, system_prompt, max_tokens)

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _build_url(provider, settings.rewrite_model, settings),
            headers=_build_headers(provider, settings),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    text, usage_raw = _parse_response(provider, data)
    return text, LLMUsage(
        model=settings.rewrite_model,
        prompt_tokens=usage_raw["input_tokens"],
        completion_tokens=usage_raw["output_tokens"],
    )


_GROUNDING_SYSTEM_PROMPT = """\
검색된 컨텍스트가 사용자 질문에 답변 가능한지 평가하세요.

질문 유형을 분석하여 아래 규칙에 따라 'is_groundable' 여부를 판정하세요:

1. [일상 잡담 및 인사]
   - 질문이 일상적인 인사(안녕하세요), 감사 표현(감사합니다), 어시스턴트 정체성 확인(너는 누구니) 등인 경우:
     * 컨텍스트 내용에 상관없이 "is_groundable"을 true로 설정하세요.

2. [범용 기술/일반 지식 질문]
   - 질문이 특정 프로젝트나 워크스페이스에 종속되지 않는 범용적인 지식(예: "JWT가 무엇인가요?", "FastAPI Dependency Injection 사용법", "SQL JOIN 문법")인 경우:
     * 컨텍스트 내용에 상관없이 "is_groundable"을 true로 설정하세요.

3. [프로젝트/워크스페이스 고유 질문]
   - 질문이 사내 프로젝트 소스 코드, 특정 데이터베이스 테이블 스키마, 특정 문서 등 워크스페이스 내부 정보에 의존하는 경우:
     * 컨텍스트 내에 질문에 답할 수 있는 명확한 근거가 존재하는 경우에만 "is_groundable"을 true로 설정하세요.
     * 컨텍스트에 관련 근거가 전혀 없는 경우 "is_groundable"을 false로 설정하세요.

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
{"is_groundable": true/false, "confidence": 0.0~1.0}

[Few-Shot 판정 예시]
1. 질문: "반가워요! 너는 이름이 뭐야?"
   판정: {"is_groundable": true, "confidence": 1.0}

2. 질문: "Java 21 버전의 가상 스레드(Virtual Thread)에 대해 설명해줘."
   판정: {"is_groundable": true, "confidence": 1.0}

3. 질문: "우리 회사 결제 시스템에서 사용하는 배치의 실행 스케줄 명세서 보여줘."
   - 검색된 컨텍스트에 관련 명세서나 배치 스케줄에 관한 파일/내용이 존재할 때:
     판정: {"is_groundable": true, "confidence": 0.9}
   - 검색된 컨텍스트에 관련 내용이 전혀 없을 때:
     판정: {"is_groundable": false, "confidence": 0.0}
"""

async def call_grounding(prompt: str) -> tuple[dict, LLMUsage]:
    """GROUNDING_MODEL(gemini-*)로 그라운딩 이진 판정을 수행하고 파싱된 dict를 반환합니다.

    Args:
        prompt: "질문: ...\n\n검색된 컨텍스트:\n..." 형식의 판정 입력.

    Returns:
        ({"is_groundable": bool, "confidence": float}, LLMUsage).
        파싱 실패 시 ({"is_groundable": False, "confidence": 0.0}, usage) 반환.
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
        _GROUNDING_SYSTEM_PROMPT,
        max_tokens=64,
        response_schema=grounding_schema,
    )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            _build_url(provider, settings.grounding_model, settings),
            headers=_build_headers(provider, settings),
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()

    text, usage_raw = _parse_response(provider, data)
    text = text.strip()
    usage = LLMUsage(
        model=settings.grounding_model,
        prompt_tokens=usage_raw["input_tokens"],
        completion_tokens=usage_raw["output_tokens"],
    )

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        return json.loads(text), usage
    except json.JSONDecodeError:
        logger.warning("call_grounding: JSON 파싱 실패. 응답: %r", text)
        return {}, usage


async def call_structured(
    messages: list[dict],
    system_prompt: str,
    max_tokens: int = 1024,
) -> dict:
    """MAIN_MODEL로 JSON 구조화 응답을 생성합니다.

    system_prompt에서 반드시 JSON 형식 응답을 명시해야 합니다.
    마크다운 코드 블록(```json...```)이 포함된 경우 자동으로 제거합니다.
    """
    text, _ = await call_main(messages, system_prompt, max_tokens=max_tokens)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)

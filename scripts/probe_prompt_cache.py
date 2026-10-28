"""
GMS 프록시 경유 Anthropic 프롬프트 캐싱 지원 확인 프로브 스크립트 (E2).

system 블록에 cache_control(ephemeral)을 붙인 동일 요청을 2회 연속 보내고,
각 응답의 usage 키/값(cache_read_input_tokens, cache_creation_input_tokens 등)을
출력해 GMS 프록시가 cache_control과 캐시 usage 필드를 통과시키는지 1회성으로 확인한다.
URL/헤더는 provider.py의 _build_url/_build_headers를 그대로 재사용한다.

주의:
- 이 스크립트는 실제 GMS API를 호출한다. RUN_LIVE_LLM=1 과 settings.gms_api_key가
  모두 있어야 실행되며, 아니면 안내 후 종료 코드 2로 끝난다. import 시에는 호출하지 않는다.
- 시스템 텍스트는 가상 사내 규정 문구의 반복이며 PII를 포함하지 않는다.
- 응답 본문 텍스트는 절대 출력하지 않는다. status_code, stop_reason, usage 키/값만 출력한다.
- 설정은 app.config.get_settings()로만 읽으며 .env를 직접 읽지 않는다.
- 최소 캐시 가능 길이: claude-sonnet-4-6 1,024토큰, claude-haiku-4-5 4,096토큰.

사용법:
    PYTHONIOENCODING=utf-8 RUN_LIVE_LLM=1 uv run python scripts/probe_prompt_cache.py
    PYTHONIOENCODING=utf-8 RUN_LIVE_LLM=1 uv run python scripts/probe_prompt_cache.py --model claude-haiku-4-5-20251001
"""

import argparse
import asyncio
import os
import sys

import httpx

from app.config import get_settings
from app.core.llm import provider

_USER_MESSAGE = "한 단어로 답하세요: 안녕"

# 가상 사내 규정 안내문 블록(PII 없음). 반복해서 최소 캐시 길이를 넘긴다.
_GUIDELINE_BLOCK = (
    "[가상 사내 업무 지침] 모든 구성원은 요청서를 작성할 때 목적, 범위, 기한을 명확히 기재해야 합니다. "
    "검토자는 접수 후 영업일 기준 이틀 안에 승인 또는 보완 요청을 회신합니다. "
    "보완 요청 시에는 부족한 항목과 기대하는 수준을 구체적으로 적어 재작업을 줄입니다. "
    "회의록은 회의 종료 당일 공유하고, 결정 사항과 담당자, 후속 일정을 표로 정리합니다. "
    "장애가 발생하면 영향 범위와 임시 조치를 먼저 공유하고, 원인 분석은 안정화 이후 작성합니다. "
)

_MIN_CHARS_DEFAULT = 3200  # 약 1,500토큰 (sonnet 최소 1,024토큰 초과)
_MIN_CHARS_HAIKU = 14000  # haiku 최소 캐시 4,096토큰 초과


def _build_system_text(model: str) -> str:
    """모델별 최소 캐시 길이를 넘도록 고정 지침 블록을 반복해 시스템 텍스트를 만든다."""
    min_chars = _MIN_CHARS_HAIKU if model.startswith("claude-haiku") else _MIN_CHARS_DEFAULT
    repeat = -(-min_chars // len(_GUIDELINE_BLOCK))
    return _GUIDELINE_BLOCK * repeat


def _build_request_body(model: str, system_text: str) -> dict:
    return {
        "model": model,
        "max_tokens": 32,
        "system": [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": _USER_MESSAGE}],
    }


async def _send_once(
    client: httpx.AsyncClient, url: str, headers: dict, body: dict, label: str
) -> dict | None:
    """요청 1회를 보내고 status_code, stop_reason, usage 키/값만 출력한다. usage dict를 반환한다."""
    resp = await client.post(url, headers=headers, json=body)
    print(f"[{label}] status_code: {resp.status_code}")
    try:
        data = resp.json()
    except ValueError:
        print("  (응답 본문이 JSON이 아님)")
        return None
    if not isinstance(data, dict):
        print(f"  (dict 아님, type={type(data).__name__})")
        return None

    print(f"  stop_reason: {data.get('stop_reason')}")
    usage = data.get("usage")
    if not isinstance(usage, dict):
        print("  usage: (없음)")
        return None
    for key, value in usage.items():
        print(f"  usage.{key}: {value}")
    return usage


def _verdict(second_usage: dict | None) -> str:
    """2회차 응답 usage로 캐싱 동작 여부를 판정한다."""
    if not second_usage or "cache_read_input_tokens" not in second_usage:
        return "GMS가 usage 캐시 필드를 전달하지 않음"
    if (second_usage.get("cache_read_input_tokens") or 0) > 0:
        return "GMS 경유 캐싱 동작"
    if (second_usage.get("cache_creation_input_tokens") or 0) > 0:
        return "캐시 기록은 되지만 재사용 안 됨(cache_control은 통과, 2회차도 cache_creation만 발생)"
    return "캐싱 미적용(프록시가 cache_control을 버렸거나 버전 헤더 미지원)"


async def _main(model: str) -> None:
    settings = get_settings()
    url = provider._build_url("anthropic", model, settings)
    headers = provider._build_headers("anthropic", settings)
    system_text = _build_system_text(model)
    body = _build_request_body(model, system_text)

    print(f"model: {model}")
    print(f"system_text_chars: {len(system_text)}")
    print(f"anthropic-version: {provider._ANTHROPIC_VERSION}")
    print(f"POST {url}")

    async with httpx.AsyncClient(timeout=60.0) as client:
        await _send_once(client, url, headers, body, "1회차")
        second = await _send_once(client, url, headers, body, "2회차")

    print(f"\n판정: {_verdict(second)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GMS 프롬프트 캐싱 지원 확인")
    parser.add_argument("--model", default=None, help="기본값: settings.main_model")
    args = parser.parse_args()

    settings = get_settings()
    if os.environ.get("RUN_LIVE_LLM") != "1" or not settings.gms_api_key:
        print("RUN_LIVE_LLM=1 과 GMS_API_KEY(settings.gms_api_key)가 모두 필요합니다. 실행하지 않습니다.")
        return 2

    asyncio.run(_main(args.model or settings.main_model))
    return 0


if __name__ == "__main__":
    sys.exit(main())

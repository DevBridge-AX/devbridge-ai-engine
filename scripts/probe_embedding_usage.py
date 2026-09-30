"""
Gemini Embedding API 응답 구조 프로브 스크립트 (A7 - usage_logs 토큰 집계 조사용).

embedder.py의 EmbedResult.total_tokens는 실측 토큰이 아니라
len(texts) * embedding_tokens_per_text로 계산한 추정치다. 이 스크립트는
batchEmbedContents 응답에 usageMetadata(실측 토큰 필드)가 실제로 존재하는지,
그리고 별도 countTokens 엔드포인트로 실측 토큰 수를 얻을 수 있는지 1회성으로
확인하기 위한 조사용 스크립트다. embedder.py와 동일한 URL/헤더 구성을 사용한다.

주의:
- 이 스크립트는 실제 GMS API를 호출한다(gms_api_key 필요). 호출 자체가
  usage_logs 등 이 레포의 누적 로직에 영향을 주지는 않는다.
- 응답 원문(임베딩 벡터, 텍스트 등 값)은 절대 출력하지 않는다. 최상위 키 구조,
  usageMetadata 존재 여부, countTokens의 상태 코드/최상위 키만 출력한다.
- 설정은 app.config.get_settings()로만 읽으며 .env를 직접 읽지 않는다.

조사 결과 (2026-09-30):
- batchEmbedContents 응답에는 최상위 usageMetadata가 존재하며, 배치(요청) 1건당
  promptTokenCount 1개를 반환한다(텍스트 건별 값이 아님). 이 값은
  embedder.embed_texts()에서 배치별로 합산되어 EmbedResult.provider_tokens로
  노출되고, 인덱싱 메트릭(embedding_provider_tokens)에만 기록된다.
- countTokens 엔드포인트도 정상 동작하지만, 별도 API 호출이 추가로 필요해
  요청 수가 늘어나므로 채택하지 않았다.

사용법:
    python3 scripts/probe_embedding_usage.py
"""

import asyncio

import httpx

from app.config import get_settings

# 실제 문서/커밋 내용이 아닌 더미 텍스트만 사용한다.
_DUMMY_TEXTS = ["probe dummy text one", "probe dummy text two"]


def _print_key_structure(data: object, label: str) -> None:
    """dict의 최상위 키와, 리스트 값인 경우 첫 item의 키만 출력한다(값/벡터 미포함)."""
    if not isinstance(data, dict):
        print(f"  {label}: (dict 아님, type={type(data).__name__})")
        return

    top_keys = sorted(data.keys())
    print(f"  {label} top-level keys: {top_keys}")

    for key in top_keys:
        value = data[key]
        if isinstance(value, list) and value and isinstance(value[0], dict):
            item_keys = sorted(value[0].keys())
            print(f"  {label}.{key}[0] keys: {item_keys}")


async def _probe_batch_embed_contents(settings) -> None:
    """embedder.embed_texts()와 동일한 방식으로 batchEmbedContents를 1회 호출한다."""
    url = f"{settings.gemini_base_url}/models/{settings.embedding_model}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.gms_api_key,
    }
    model_path = f"models/{settings.embedding_model}"
    body = {
        "requests": [
            {"model": model_path, "content": {"parts": [{"text": t}]}}
            for t in _DUMMY_TEXTS
        ]
    }

    print(f"[1] POST {url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=body)

    print(f"  status_code: {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        print("  (응답 본문이 JSON이 아님)")
        return

    _print_key_structure(data, "response")
    usage_present = isinstance(data, dict) and "usageMetadata" in data
    print(f"  usageMetadata present: {usage_present}")


async def _probe_count_tokens(settings) -> None:
    """models/{embedding_model}:countTokens를 1회 호출해 상태 코드/최상위 키만 확인한다."""
    url = f"{settings.gemini_base_url}/models/{settings.embedding_model}:countTokens"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.gms_api_key,
    }
    body = {"contents": [{"parts": [{"text": t}]} for t in _DUMMY_TEXTS]}

    print(f"\n[2] POST {url}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, headers=headers, json=body)

    print(f"  status_code: {resp.status_code}")

    try:
        data = resp.json()
    except ValueError:
        print("  (응답 본문이 JSON이 아님)")
        return

    _print_key_structure(data, "response")


async def _main() -> None:
    settings = get_settings()

    if not settings.gms_api_key:
        print("gms_api_key가 비어 있습니다. .env에 GMS_API_KEY를 설정한 뒤 다시 실행하세요.")
        return

    await _probe_batch_embed_contents(settings)
    await _probe_count_tokens(settings)


if __name__ == "__main__":
    asyncio.run(_main())

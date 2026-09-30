"""
임베딩 생성 및 모델 버전 관리.

Google Gemini Embedding API (GMS 프록시 경유)의 batchEmbedContents 엔드포인트를
사용합니다. 엔드포인트 URL은 config의 gemini_base_url + /models/ + embedding_model로
동적으로 구성됩니다.

content(원문)는 불변이며, 모델 교체 시 이 모듈만 재호출하면 됩니다
(content 재사용, embedding만 재생성).

주의:
- Gemini 임베딩 API는 응답에 실측 토큰 수(usageMetadata 등)를 포함하지 않습니다.
  EmbedResult.total_tokens는 실측값이 아니라
  `len(texts) * settings.embedding_tokens_per_text`(기본 0.2, GMS 과금 단위
  추정치)로 계산한 추정치이며, EmbedResult.token_source는 항상
  "estimate_per_text"입니다. 즉 usage_logs.embedding_tokens에 누적되는 값도
  실 토큰 집계가 아닌 추정치입니다. 추후 Google/GMS API가 실측 토큰 필드를
  제공하면 이 부분을 업데이트하세요.
"""

from dataclasses import dataclass

import httpx

from app.config import get_settings

_BATCH_SIZE = 100  # batchEmbedContents 최대 100건


@dataclass
class EmbedResult:
    embeddings: list[list[float]]
    embedding_model: str
    embedding_model_version: str
    total_tokens: float
    # 실측 아님: len(texts) * settings.embedding_tokens_per_text로 계산한 추정치.
    # 추후 실측 토큰원이 추가되면 이 값도 함께 갱신해야 합니다.
    token_source: str = "estimate_per_text"


async def embed_texts(texts: list[str]) -> EmbedResult:
    """텍스트 목록에 대한 임베딩 벡터를 생성합니다.

    100건 초과 시 내부적으로 배치 분할하여 처리합니다.
    """
    if not texts:
        raise ValueError("embed_texts: texts must not be empty")

    settings = get_settings()
    url = f"{settings.gemini_base_url}/models/{settings.embedding_model}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.gms_api_key,
    }
    model_path = f"models/{settings.embedding_model}"

    all_embeddings: list[list[float]] = []

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        body = {
            "requests": [
                {"model": model_path, "content": {"parts": [{"text": t}]}}
                for t in batch
            ]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        all_embeddings.extend(item["values"] for item in data["embeddings"])

    return EmbedResult(
        embeddings=all_embeddings,
        embedding_model=settings.embedding_model,
        embedding_model_version=settings.embedding_model_version,
        total_tokens=len(texts) * settings.embedding_tokens_per_text,
    )
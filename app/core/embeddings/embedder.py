"""
임베딩 생성 및 모델 버전 관리.

config.embedding_model(기본값: text-embedding-3-large)로 OpenAI Embeddings API를
호출합니다. content(원문)는 불변이며, 모델 교체 시 이 모듈만 재호출하면 됩니다
(content 재사용, embedding만 재생성).

확인 필요:
- EMBEDDING_API_KEY는 OpenAI API 키를 가정합니다. 다른 임베딩 제공자를 사용하는
  경우 _API_URL 및 요청/응답 파싱 로직을 조정해야 합니다.
"""

from dataclasses import dataclass

import httpx

from app.config import get_settings

_API_URL = "https://api.openai.com/v1/embeddings"  # TODO: 설정에서 불러오기
_BATCH_SIZE = 100  # [설정] OpenAI 최대 2048이지만 안전 마진을 두어 100으로 제한


@dataclass
class EmbedResult:
    embeddings: list[list[float]]
    embedding_model: str
    embedding_model_version: str
    total_tokens: int


async def embed_texts(texts: list[str]) -> EmbedResult:
    """텍스트 목록에 대한 임베딩 벡터를 생성합니다.

    100건 초과 시 내부적으로 배치 분할하여 처리합니다. 반환값의 total_tokens는
    usage_logs 누적에 사용합니다.
    """
    if not texts:
        raise ValueError("embed_texts: texts must not be empty")

    settings = get_settings()
    all_embeddings: list[list[float]] = []
    total_tokens = 0

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                _API_URL,
                headers={"Authorization": f"Bearer {settings.embedding_api_key}"},
                json={"model": settings.embedding_model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()

        items = sorted(data["data"], key=lambda x: x["index"])
        all_embeddings.extend(item["embedding"] for item in items)
        total_tokens += data["usage"]["total_tokens"]

    return EmbedResult(
        embeddings=all_embeddings,
        embedding_model=settings.embedding_model,
        embedding_model_version=settings.embedding_model_version,
        total_tokens=total_tokens,
    )

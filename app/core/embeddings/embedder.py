"""
임베딩 생성 및 모델 버전 관리.

Google Gemini Embedding API (GMS 프록시 경유)의 batchEmbedContents 엔드포인트를
사용합니다. 엔드포인트 URL은 config의 gemini_base_url + /models/ + embedding_model로
동적으로 구성됩니다.

content(원문)는 불변이며, 모델 교체 시 이 모듈만 재호출하면 됩니다
(content 재사용, embedding만 재생성).

토큰 필드 정리 (2026-09-30 조사 결과 반영):
- EmbedResult.total_tokens: usage_logs에 사용되는 과금 단위 추정치입니다.
  `len(texts) * settings.embedding_tokens_per_text`(기본 0.2, GMS 과금 단위
  추정치)로 계산하며, token_source는 항상 "estimate_per_text"입니다(이 필드가
  total_tokens의 산출 방식을 설명합니다). usage_logs.embedding_tokens의 의미
  (provider 토큰인지 GMS 과금 단위인지)는 Spring 쪽 결정이 아직 열려 있어,
  이 값은 그대로 유지합니다.
- EmbedResult.provider_tokens: batchEmbedContents 응답의
  usageMetadata.promptTokenCount를 배치 단위로 합산한 실측값입니다(요청 1건당
  1개 카운트이며 텍스트 건별 값이 아님). 메트릭(embedding_provider_tokens)
  기록에만 사용되며, usage_logs에는 반영하지 않습니다. 배치 중 하나라도
  usageMetadata가 없으면 부분합은 의미가 없으므로 전체를 None으로 둡니다.

taskType (선택):
- embed_texts(task_type=...)를 주면 각 요청 객체에 "taskType"을 추가합니다. None(기본)이면
  요청 본문에 taskType 필드를 넣지 않아 기존 동작과 동일합니다.
- 질의 측 호출부는 query_embed_kwargs()로 settings.embedding_query_task_type을 전달합니다
  (설정이 None이면 인자를 아예 넘기지 않음). 문서 인덱싱 호출부는 taskType을 쓰지 않습니다.
"""

from dataclasses import dataclass

import httpx

from app.config import EMBEDDING_TASK_TYPES, EmbeddingTaskType, get_settings

__all__ = [
    "EMBEDDING_TASK_TYPES",
    "EmbeddingTaskType",
    "EmbedResult",
    "embed_texts",
    "query_embed_kwargs",
]

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
    # 실측: batchEmbedContents 응답 usageMetadata.promptTokenCount를 배치별로
    # 합산한 값(요청 1건당 1개 카운트). 배치 중 하나라도 값이 없으면 None.
    provider_tokens: int | None = None
    # 요청에 사용한 taskType(없으면 None = 요청 본문에 taskType 미포함).
    task_type: str | None = None


def query_embed_kwargs() -> dict[str, str]:
    """질의 측 embed_texts 호출용 kwargs. 설정이 None이면 빈 dict(인자를 넘기지 않음)."""
    task_type = get_settings().embedding_query_task_type
    return {} if task_type is None else {"task_type": task_type}


async def embed_texts(texts: list[str], task_type: str | None = None) -> EmbedResult:
    """텍스트 목록에 대한 임베딩 벡터를 생성합니다.

    100건 초과 시 내부적으로 배치 분할하여 처리합니다.
    task_type이 주어지면 각 요청에 taskType을 추가하며, 허용 값이 아니면 ValueError입니다.
    """
    if not texts:
        raise ValueError("embed_texts: texts must not be empty")
    if task_type is not None and task_type not in EMBEDDING_TASK_TYPES:
        raise ValueError(
            f"embed_texts: unknown task_type {task_type!r} (allowed: {', '.join(EMBEDDING_TASK_TYPES)})"
        )

    settings = get_settings()
    url = f"{settings.gemini_base_url}/models/{settings.embedding_model}:batchEmbedContents"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": settings.gms_api_key,
    }
    model_path = f"models/{settings.embedding_model}"

    all_embeddings: list[list[float]] = []
    provider_tokens_sum = 0
    provider_tokens_complete = True

    for i in range(0, len(texts), _BATCH_SIZE):
        batch = texts[i : i + _BATCH_SIZE]
        body = {
            "requests": [
                {
                    "model": model_path,
                    "content": {"parts": [{"text": t}]},
                    **({"taskType": task_type} if task_type is not None else {}),
                }
                for t in batch
            ]
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

        all_embeddings.extend(item["values"] for item in data["embeddings"])

        # batchEmbedContents는 요청(배치) 1건당 usageMetadata 1개를 반환한다
        # (텍스트 건별 값이 아님). 배치 중 하나라도 없으면 부분합이 오도되므로
        # provider_tokens 전체를 None으로 처리한다.
        batch_prompt_tokens = data.get("usageMetadata", {}).get("promptTokenCount")
        if batch_prompt_tokens is None:
            provider_tokens_complete = False
        elif provider_tokens_complete:
            provider_tokens_sum += batch_prompt_tokens

    return EmbedResult(
        embeddings=all_embeddings,
        embedding_model=settings.embedding_model,
        embedding_model_version=settings.embedding_model_version,
        total_tokens=len(texts) * settings.embedding_tokens_per_text,
        provider_tokens=provider_tokens_sum if provider_tokens_complete else None,
        task_type=task_type,
    )
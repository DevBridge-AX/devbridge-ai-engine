"""
LLM 리랭커 (listwise).

retriever가 가져온 후보 K건을 번호 목록으로 경량 LLM에 주고, 질문에 가장 유용한 순서의
번호 목록(JSON)을 받아 상위 top_k건으로 줄입니다. 기본 off(rerank_enabled=False).

원칙:
- 순서만 바꿉니다. RetrievedChunk 객체와 similarity_score는 그대로이므로 citations와
  그라운딩 임계치의 의미가 변하지 않습니다.
- fail-open: 타임아웃·HTTP 오류·응답 파싱 실패는 모두 원래 순서의 상위 top_k로 대체하고,
  사유를 RerankResult.stage("fallback:<reason>")에 남깁니다.
- 토큰 사용량은 RerankResult.usage로만 돌려주며 Spring 응답(token_usage)에는 포함하지 않습니다.
  호출 1건은 llm_calls(purpose="rerank")에도 기록됩니다.
- 후보는 retriever의 접근 제어 필터를 이미 통과한 결과이므로 여기서 추가 필터는 하지 않습니다.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import httpx

from app.config import get_settings
from app.core.llm.provider import LLMUsage, call_with_model
from app.core.rag.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_RERANK_MAX_TOKENS = 64

_RERANK_SCHEMA = {
    "type": "OBJECT",
    "properties": {"ranking": {"type": "ARRAY", "items": {"type": "INTEGER"}}},
    "required": ["ranking"],
}


@dataclass
class RerankResult:
    chunks: list[RetrievedChunk]  # 길이 <= top_k
    applied: bool  # True면 LLM 순서, False면 원래 순서
    stage: str  # "applied" | "disabled" | "skipped_few" | "fallback:<reason>" (예: fallback:timeout, fallback:http_429)
    latency_ms: float
    usage: LLMUsage | None


def build_rerank_prompt(query: str, chunks: list[RetrievedChunk], top_k: int) -> tuple[str, str]:
    """(system_prompt, user_prompt)를 만듭니다."""
    max_chars = get_settings().rerank_max_chunk_chars
    system = (
        "너는 질문에 답하는 데 가장 유용한 문단을 고르는 역할이다. "
        f"관련도가 높은 순서대로 상위 {top_k}개 후보의 번호만 골라 "
        '{"ranking": [번호, ...]} 형식의 JSON으로만 답하라.'
    )
    lines = [f"질문: {query}", "", "후보:"]
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] ({chunk.title}) {chunk.content[:max_chars]}")
    return system, "\n".join(lines)


def parse_ranking(text: str, n_candidates: int, top_k: int) -> list[int] | None:
    """LLM 응답에서 0-based 후보 인덱스 순서를 추출합니다.

    {"ranking": [...]} 또는 배열 자체를 허용합니다. 번호는 1-based로 해석하며 범위 밖·중복은
    버립니다. 결과가 top_k 미만이면 원래 순서에서 빠진 후보를 뒤에 채웁니다.
    JSON 파싱 자체가 실패하거나 유효한 번호가 하나도 없으면 None.
    """
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(data, dict):
        data = data.get("ranking")
    if not isinstance(data, list):
        return None

    order: list[int] = []
    seen: set[int] = set()
    for raw in data:
        if isinstance(raw, bool) or not isinstance(raw, int):
            continue
        idx = raw - 1
        if 0 <= idx < n_candidates and idx not in seen:
            seen.add(idx)
            order.append(idx)
    if not order:
        return None

    target = min(top_k, n_candidates)
    for idx in range(n_candidates):
        if len(order) >= target:
            break
        if idx not in seen:
            seen.add(idx)
            order.append(idx)
    return order[:top_k]


async def rerank(query: str, chunks: list[RetrievedChunk], *, top_k: int) -> RerankResult:
    """후보 chunks를 질문 관련도 순으로 재정렬해 상위 top_k건을 반환합니다."""
    settings = get_settings()
    original = chunks[:top_k]
    if not settings.rerank_enabled:
        return RerankResult(original, False, "disabled", 0.0, None)
    if len(chunks) <= top_k:
        return RerankResult(original, False, "skipped_few", 0.0, None)

    start = time.perf_counter()

    def _fallback(reason: str, usage: LLMUsage | None = None) -> RerankResult:
        return RerankResult(
            original, False, f"fallback:{reason}", (time.perf_counter() - start) * 1000, usage
        )

    system, user = build_rerank_prompt(query, chunks, top_k)
    try:
        text, usage = await asyncio.wait_for(
            call_with_model(
                settings.rerank_model,
                [{"role": "user", "content": user}],
                system,
                max_tokens=_RERANK_MAX_TOKENS,
                purpose="rerank",
                response_schema=_RERANK_SCHEMA,
            ),
            timeout=settings.rerank_timeout_seconds,
        )
    except asyncio.TimeoutError:
        logger.warning("rerank: 타임아웃, 원래 순서 사용")
        return _fallback("timeout")
    except httpx.HTTPStatusError as exc:
        # 429(분당 요청 제한) 등 상태 코드를 사유에 남겨 chat_metrics에서 원인을 구분할 수 있게 한다.
        reason = f"http_{exc.response.status_code}"
        logger.warning("rerank: 호출 실패(%s), 원래 순서 사용", reason)
        return _fallback(reason)
    except Exception as exc:  # CancelledError는 BaseException이라 그대로 전파된다
        logger.warning("rerank: 호출 실패(%s), 원래 순서 사용", type(exc).__name__)
        return _fallback(type(exc).__name__)

    order = parse_ranking(text, len(chunks), top_k)
    if order is None:
        logger.warning("rerank: 응답 파싱 실패, 원래 순서 사용")
        return _fallback("parse", usage)

    return RerankResult(
        [chunks[i] for i in order],
        True,
        "applied",
        (time.perf_counter() - start) * 1000,
        usage,
    )

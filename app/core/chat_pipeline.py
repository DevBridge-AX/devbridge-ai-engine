"""
챗봇 응답 생성 파이프라인.

run()은 ChatRequest를 받아 ChatEvent(token/done)를 순서대로 yield합니다.
LangGraph 전환 없이 단순 파이프라인 함수로 구현합니다.

흐름:
  truncate_history → query_rewriter → [semantic_cache 조회] → retriever → grounding
    → [cache hit] 저장된 답변을 token* → done 으로 재생 (그라운딩·메인 LLM 생략)
    → [실패] done(is_groundable=False)
    → [통과] persona_prompt + call_main_stream → token* → [첫 턴이면 cache 저장] → done

시맨틱 캐시(semantic_cache_enabled, 기본 off): 켜지면 rewritten_query 임베딩을 한 번만
계산해 캐시 조회와 retriever 양쪽에서 재사용합니다. 꺼져 있으면 기존 흐름과 동일합니다.
hit 시에는 접근 제어를 재검증하고, 저장은 첫 턴 + 정상 그라운딩 + 비어있지 않은 답변일
때만 수행합니다. semantic_cache_verify_enabled를 켜면 임계치 아래 후보(최선 1건)를 경량 LLM으로
재검증해 YES일 때만 hit로 인정합니다(접근 재검증 후에 호출, 실패/타임아웃은 miss, 검증 호출의
토큰은 token_usage에 넣지 않고 llm_calls의 purpose="cache_verify"로만 기록). 응답 형식(token/done)은 변하지 않으며 hit의 token_usage.main은 0입니다.

관측성: 요청 1건당 구간별 소요 시간(rewrite_ms/retrieve_ms/grounding_ms/llm_ms/
ttft_ms/total_ms)과 품질 신호(chunk_count/top_similarity/is_groundable/confidence 등)를
app.core.metrics.record_metric으로 기록합니다(조기 종료·정상 경로 모두 1건).
질문/답변 원문·user_id는 기록하지 않으며, session_id는 SHA-256 해시 앞 12자리만 남깁니다.
"""

import hashlib
import logging
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.cache import verifier
from app.core.cache.semantic_cache import CacheEntry, get_semantic_cache, make_namespace
from app.core.embeddings.embedder import embed_texts
from app.core.llm import provider as llm
from app.core.llm import query_rewriter
from app.core.llm.persona_prompts import get_system_prompt
from app.core.llm.provider import LLMUsage
from app.core.metrics import StageTimer, record_metric
from app.core.rag import grounding, retriever
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import AccessFilter, RetrievedChunk
from app.core.utils.token_counter import truncate_history
from app.schemas.chat import (
    ChatDoneEvent,
    ChatRequest,
    Citation,
    TokenUsage,
    TokenUsageDetail,
)

logger = logging.getLogger(__name__)

PROMPT_VERSION = "persona-v1"

# 히스토리 토큰 예산 (max_context_tokens 30000 기준)
# 시스템 프롬프트 ~800 + RAG 컨텍스트 ~4000 + 현재 질문 ~200 + 응답 예약 ~3000 = ~8000
# 히스토리 가용 = 30000 - 8000 = 22000
_HISTORY_TOKEN_BUDGET = 22_000


@dataclass
class ChatEvent:
    event: str  # "token" | "done" | "error"
    data: dict


async def run(request: ChatRequest, db: Session) -> AsyncGenerator[ChatEvent, None]:
    """ChatRequest를 처리하고 SSE 이벤트(ChatEvent)를 순서대로 yield합니다."""
    timer = StageTimer()
    total_start = time.perf_counter()

    history_dicts = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    truncated_history, context_truncated = truncate_history(history_dicts, budget=_HISTORY_TOKEN_BUDGET)

    is_first_turn = len(truncated_history) == 0
    turn = len(history_dicts) // 2 + 1

    if is_first_turn:
        rewritten_query = request.content
        rewrite_usage: LLMUsage | None = None
    else:
        with timer.measure("rewrite_ms"):
            rewritten_query, rewrite_usage = await query_rewriter.rewrite(request.content, truncated_history)

    access = AccessFilter(
        accessible_task_ids=request.accessible_task_ids,
        can_view_restricted=request.can_view_restricted,
    )
    access_filtered = not access.is_unrestricted

    settings = get_settings()
    cache_enabled = bool(getattr(settings, "semantic_cache_enabled", False))
    query_embedding: list[float] | None = None
    cache_hit = False
    cache_similarity: float | None = None
    cache_lookup_ms: float | None = None
    cache_verify_ms: float | None = None
    cache_verify_result: str | None = None
    cache = None
    ns = None
    cache_generation = 0

    if cache_enabled:
        cache = get_semantic_cache()
        with timer.measure("embed_ms"):
            query_embedding = (await embed_texts([rewritten_query])).embeddings[0]
        ns = make_namespace(
            request.workspace_id,
            request.role,
            request.accessible_task_ids,
            request.can_view_restricted,
            PROMPT_VERSION,
            settings.main_model,
        )
        # 스트림 도중 인덱싱 무효화가 일어나면 stale 답변을 저장하지 않도록 세대를 잡아 둔다.
        cache_generation = cache.generation(request.workspace_id)
        lookup_start = time.perf_counter()
        # LLM 재검증이 켜져 있으면 후보 하한(candidate_threshold)까지 내려서 조회한다.
        # candidate_threshold > threshold면 후보 구간이 없으므로 재검증을 쓰지 않는다.
        verify_enabled = bool(getattr(settings, "semantic_cache_verify_enabled", False))
        lookup_floor: float | None = None
        if verify_enabled:
            candidate_threshold = settings.semantic_cache_candidate_threshold
            if candidate_threshold <= cache.threshold:
                lookup_floor = candidate_threshold
            else:
                logger.warning(
                    "semantic cache verify skipped: candidate_threshold(%s) > threshold(%s)",
                    candidate_threshold,
                    cache.threshold,
                )
        found = cache.lookup(ns, query_embedding, min_similarity=lookup_floor)
        cache_lookup_ms = (time.perf_counter() - lookup_start) * 1000
        if found is not None:
            entry, similarity = found
            # 접근 제어 재검증: 캐시된 근거 청크가 지금도 이 요청의 접근 범위에 있어야 한다.
            # (유료 LLM 재검증보다 먼저 수행한다.)
            cited_ids = {c["chunk_id"] for c in entry.citations}
            access_ok = not (cited_ids and retriever._allowed_chunk_ids(cited_ids, access, db) != cited_ids)
            if not access_ok:
                cache.remove(ns, entry)
            verified = access_ok
            if access_ok and similarity < cache.threshold:
                # 후보 구간: 최선 후보 1건만 LLM으로 "같은 질문인가" 확인한다. YES만 hit.
                with timer.measure("cache_verify_ms"):
                    verdict = await verifier.verify_same_question(rewritten_query, entry.query)
                verified = verdict.same
                cache_verify_ms = verdict.latency_ms
                cache_verify_result = verdict.outcome
                if verified:
                    cache.mark_hit(ns, entry)
            if verified:
                cache_hit = True
                cache_similarity = similarity
                first_token_at: float | None = None
                for piece in _split_answer(entry.answer):
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    yield ChatEvent(event="token", data={"text": piece})
                ttft_ms = (first_token_at - total_start) * 1000 if first_token_at is not None else None
                token_usage = TokenUsage(
                    main=TokenUsageDetail(model=settings.main_model, prompt_tokens=0, completion_tokens=0),
                    rewrite=_to_usage_detail(rewrite_usage) if rewrite_usage else None,
                    grounding=None,
                    context_truncated=context_truncated,
                )
                cached_citations = [Citation(**c) for c in entry.citations]
                _record_chat_metric(
                    request=request,
                    timer=timer,
                    total_start=total_start,
                    ttft_ms=ttft_ms,
                    turn=turn,
                    chunk_count=len(cached_citations),
                    top_similarity=max((c.similarity_score for c in cached_citations), default=0.0),
                    grounding_result=GroundingResult(is_groundable=True, confidence=entry.confidence),
                    grounding_stage="cache",
                    access_filtered=access_filtered,
                    token_usage=token_usage,
                    cache_enabled=True,
                    cache_hit=True,
                    cache_similarity=cache_similarity,
                    cache_lookup_ms=cache_lookup_ms,
                    cache_verify_ms=cache_verify_ms,
                    cache_verify_result=cache_verify_result,
                )
                yield ChatEvent(
                    event="done",
                    data=ChatDoneEvent(
                        citations=cached_citations,
                        is_groundable=True,
                        confidence=entry.confidence,
                        suggested_owner_id=entry.suggested_owner_id,
                        prompt_version=PROMPT_VERSION,
                        token_usage=token_usage,
                    ).model_dump(),
                )
                return

    retrieve_kwargs = {"query_embedding": query_embedding} if cache_enabled else {}
    with timer.measure("retrieve_ms"):
        chunks = await retriever.retrieve(
            rewritten_query,
            request.workspace_id,
            db,
            top_k=5,
            access=access,
            timer=timer,
            **retrieve_kwargs,
        )

    with timer.measure("grounding_ms"):
        grounding_result = await grounding.assess(chunks, request.content)
    grounding_usage = grounding_result.llm_usage
    if grounding_result.fallback_reason is not None:
        grounding_stage = "llm_fallback"
    elif grounding_usage is not None:
        grounding_stage = "llm"
    else:
        grounding_stage = "threshold"

    citations = _build_citations(chunks)
    top_similarity = max((c.similarity_score for c in chunks), default=0.0)

    cache_fields = {
        "cache_enabled": cache_enabled,
        "cache_hit": False,
        "cache_similarity": None,
        "cache_lookup_ms": cache_lookup_ms,
        "cache_verify_ms": cache_verify_ms,
        "cache_verify_result": cache_verify_result,
    }

    if not grounding_result.is_groundable:
        token_usage = TokenUsage(
            main=TokenUsageDetail(
                model=settings.main_model,
                prompt_tokens=0,
                completion_tokens=0,
            ),
            rewrite=_to_usage_detail(rewrite_usage) if rewrite_usage else None,
            grounding=_to_usage_detail(grounding_usage) if grounding_usage else None,
            context_truncated=context_truncated,
        )
        _record_chat_metric(
            request=request,
            timer=timer,
            total_start=total_start,
            ttft_ms=None,
            turn=turn,
            chunk_count=len(chunks),
            top_similarity=top_similarity,
            grounding_result=grounding_result,
            grounding_stage=grounding_stage,
            access_filtered=access_filtered,
            token_usage=token_usage,
            **cache_fields,
        )
        yield ChatEvent(
            event="done",
            data=ChatDoneEvent(
                citations=citations,
                is_groundable=False,
                confidence=grounding_result.confidence,
                suggested_owner_id=grounding_result.suggested_owner_id,
                prompt_version=PROMPT_VERSION,
                token_usage=token_usage,
            ).model_dump(),
        )
        return

    retrieved_context = _format_context(chunks)
    system_prompt = get_system_prompt(request.role, retrieved_context, truncated_history)
    messages = _build_messages(truncated_history, request.content)

    main_usage: LLMUsage | None = None
    first_token_at: float | None = None
    answer_parts: list[str] = []
    with timer.measure("llm_ms"):
        async for text, usage in llm.call_main_stream(
            messages, system_prompt, max_tokens=settings.main_max_tokens
        ):
            if text is not None:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                answer_parts.append(text)
                yield ChatEvent(event="token", data={"text": text})
            else:
                main_usage = usage

    if main_usage is None:
        main_usage = LLMUsage(model=settings.main_model, prompt_tokens=0, completion_tokens=0)

    ttft_ms = (first_token_at - total_start) * 1000 if first_token_at is not None else None
    token_usage = TokenUsage(
        main=_to_usage_detail(main_usage),
        rewrite=_to_usage_detail(rewrite_usage) if rewrite_usage else None,
        grounding=_to_usage_detail(grounding_usage) if grounding_usage else None,
        context_truncated=context_truncated,
    )

    _record_chat_metric(
        request=request,
        timer=timer,
        total_start=total_start,
        ttft_ms=ttft_ms,
        turn=turn,
        chunk_count=len(chunks),
        top_similarity=top_similarity,
        grounding_result=grounding_result,
        grounding_stage=grounding_stage,
        access_filtered=access_filtered,
        token_usage=token_usage,
        **cache_fields,
    )

    answer_text = "".join(answer_parts)
    if (
        cache_enabled
        and is_first_turn
        and grounding_result.is_groundable
        and grounding_result.fallback_reason is None
        and answer_text.strip()
    ):
        if cache.generation(request.workspace_id) != cache_generation:
            logger.debug("semantic cache store skipped: workspace invalidated during generation")
        else:
            cache.store(
                ns,
                query_embedding,
                CacheEntry(
                    embedding=[],
                    query=rewritten_query,
                    answer=answer_text,
                    citations=[c.model_dump() for c in citations],
                    confidence=grounding_result.confidence,
                    suggested_owner_id=None,
                    created_at=cache.now(),
                ),
            )

    yield ChatEvent(
        event="done",
        data=ChatDoneEvent(
            citations=citations,
            is_groundable=True,
            confidence=grounding_result.confidence,
            suggested_owner_id=None,
            prompt_version=PROMPT_VERSION,
            token_usage=token_usage,
        ).model_dump(),
    )


def _record_chat_metric(
    *,
    request: ChatRequest,
    timer: StageTimer,
    total_start: float,
    ttft_ms: float | None,
    turn: int,
    chunk_count: int,
    top_similarity: float,
    grounding_result: GroundingResult,
    grounding_stage: str,
    access_filtered: bool,
    token_usage: TokenUsage,
    cache_enabled: bool = False,
    cache_hit: bool = False,
    cache_similarity: float | None = None,
    cache_lookup_ms: float | None = None,
    cache_verify_ms: float | None = None,
    cache_verify_result: str | None = None,
) -> None:
    """chat 응답 1건의 지표를 기록합니다.

    질문/답변 원문·user_id는 절대 포함하지 않으며, session_id는 SHA-256 해시 앞
    12자리만 남깁니다. 조기 종료(비그라운딩)·정상 경로 모두 이 함수로 1건 기록됩니다.
    """
    total_ms = (time.perf_counter() - total_start) * 1000
    session_id_hash = hashlib.sha256(request.session_id.encode("utf-8")).hexdigest()[:12]

    payload = {
        "session_id_hash": session_id_hash,
        "workspace_id": request.workspace_id,
        "role": request.role,
        "turn": turn,
        "rewrite_ms": timer.stages.get("rewrite_ms"),
        "retrieve_ms": timer.stages.get("retrieve_ms"),
        "embed_ms": timer.stages.get("embed_ms"),
        "bm25_ms": timer.stages.get("bm25_ms"),
        "vector_ms": timer.stages.get("vector_ms"),
        "acl_filter_ms": timer.stages.get("acl_filter_ms"),
        "grounding_ms": timer.stages.get("grounding_ms"),
        "llm_ms": timer.stages.get("llm_ms"),
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "chunk_count": chunk_count,
        "top_similarity": top_similarity,
        "is_groundable": grounding_result.is_groundable,
        "confidence": grounding_result.confidence,
        "grounding_stage": grounding_stage,
        "grounding_fallback_reason": grounding_result.fallback_reason,
        "access_filtered": access_filtered,
        "acl_refetch_count": timer.fields.get("acl_refetch_count"),
        "main_model": token_usage.main.model,
        "main_prompt_tokens": token_usage.main.prompt_tokens,
        "main_completion_tokens": token_usage.main.completion_tokens,
        "rewrite_model": token_usage.rewrite.model if token_usage.rewrite else None,
        "rewrite_prompt_tokens": token_usage.rewrite.prompt_tokens if token_usage.rewrite else None,
        "rewrite_completion_tokens": token_usage.rewrite.completion_tokens if token_usage.rewrite else None,
        "grounding_model": token_usage.grounding.model if token_usage.grounding else None,
        "grounding_prompt_tokens": token_usage.grounding.prompt_tokens if token_usage.grounding else None,
        "grounding_completion_tokens": (
            token_usage.grounding.completion_tokens if token_usage.grounding else None
        ),
        "context_truncated": token_usage.context_truncated,
        "cache_enabled": cache_enabled,
        "cache_hit": cache_hit,
        "cache_similarity": cache_similarity,
        "cache_lookup_ms": cache_lookup_ms,
        "cache_verify_ms": cache_verify_ms,
        "cache_verify_result": cache_verify_result,
    }
    record_metric("chat_metrics", payload)


def _split_answer(answer: str, size: int = 40) -> list[str]:
    """캐시된 답변을 SSE token 이벤트용 조각(기본 40자)으로 나눕니다."""
    return [answer[i : i + size] for i in range(0, len(answer), size)]


def _build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_type=chunk.source_type,
            source_id=chunk.source_id,
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            similarity_score=chunk.similarity_score,
        )
        for chunk in chunks
    ]


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] ({chunk.source_type}) {chunk.title}\n{chunk.content}")
    return "\n\n".join(lines)


def _build_messages(history: list[dict], current_query: str) -> list[dict]:
    return [*history, {"role": "user", "content": current_query}]


def _to_usage_detail(usage: LLMUsage) -> TokenUsageDetail:
    return TokenUsageDetail(
        model=usage.model,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
    )



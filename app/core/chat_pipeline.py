"""
챗봇 응답 생성 파이프라인.

run()은 ChatRequest를 받아 ChatEvent(token/done)를 순서대로 yield합니다.
LangGraph 전환 없이 단순 파이프라인 함수로 구현합니다.

흐름:
  truncate_history → query_rewriter → retriever → grounding
    → [실패] done(is_groundable=False)
    → [통과] persona_prompt + call_main_stream → token* → done

관측성: 요청 1건당 구간별 소요 시간(rewrite_ms/retrieve_ms/grounding_ms/llm_ms/
ttft_ms/total_ms)과 품질 신호(chunk_count/top_similarity/is_groundable/confidence 등)를
app.core.metrics.record_metric으로 기록합니다(조기 종료·정상 경로 모두 1건).
질문/답변 원문·user_id는 기록하지 않으며, session_id는 SHA-256 해시 앞 12자리만 남깁니다.
"""

import hashlib
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
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

    with timer.measure("retrieve_ms"):
        chunks = await retriever.retrieve(
            rewritten_query, request.workspace_id, db, top_k=5, access=access, timer=timer
        )

    with timer.measure("grounding_ms"):
        grounding_result = await grounding.assess(chunks, request.content)
    grounding_usage = grounding_result.llm_usage
    grounding_stage = "llm" if grounding_usage is not None else "threshold"

    citations = _build_citations(chunks)
    top_similarity = max((c.similarity_score for c in chunks), default=0.0)

    if not grounding_result.is_groundable:
        settings = get_settings()
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
    with timer.measure("llm_ms"):
        async for text, usage in llm.call_main_stream(messages, system_prompt):
            if text is not None:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                yield ChatEvent(event="token", data={"text": text})
            else:
                main_usage = usage

    if main_usage is None:
        settings = get_settings()
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
        "access_filtered": access_filtered,
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
    }
    record_metric("chat_metrics", payload)


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



"""
챗봇 응답 생성 파이프라인.

run()은 ChatRequest를 받아 ChatEvent(token/done)를 순서대로 yield합니다.
LangGraph 전환 없이 단순 파이프라인 함수로 구현합니다.

흐름:
  truncate_history → query_rewriter → retriever → grounding
    → [실패] done(is_groundable=False)
    → [통과] persona_prompt + call_main_stream → token* → done
"""

from collections.abc import AsyncGenerator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.config import get_settings
from app.core.llm import provider as llm
from app.core.llm import query_rewriter
from app.core.llm.persona_prompts import get_system_prompt
from app.core.llm.provider import LLMUsage
from app.core.rag import grounding, retriever
from app.core.rag.retriever import RetrievedChunk
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
    history_dicts = [{"role": m.role, "content": m.content} for m in request.conversation_history]
    truncated_history, context_truncated = truncate_history(history_dicts, budget=_HISTORY_TOKEN_BUDGET)

    is_first_turn = len(truncated_history) == 0
    if is_first_turn:
        rewritten_query = request.content
        rewrite_usage: LLMUsage | None = None
    else:
        rewritten_query, rewrite_usage = await query_rewriter.rewrite(request.content, truncated_history)

    chunks = await retriever.retrieve(rewritten_query, request.workspace_id, db, top_k=5)

    grounding_result = await grounding.assess(chunks, request.content)

    # grounding 2차 판정 토큰을 rewrite 사용량에 합산 (같은 REWRITE_MODEL)
    rewrite_usage = _merge_usage(rewrite_usage, grounding_result.llm_usage)

    citations = _build_citations(chunks)

    if not grounding_result.is_groundable:
        settings = get_settings()
        yield ChatEvent(
            event="done",
            data=ChatDoneEvent(
                citations=citations,
                is_groundable=False,
                confidence=grounding_result.confidence,
                suggested_owner_id=grounding_result.suggested_owner_id,
                prompt_version=PROMPT_VERSION,
                token_usage=TokenUsage(
                    main=TokenUsageDetail(
                        model=settings.main_model,
                        prompt_tokens=0,
                        completion_tokens=0,
                    ),
                    rewrite=_to_usage_detail(rewrite_usage) if rewrite_usage else None,
                    context_truncated=context_truncated,
                ),
            ).model_dump(),
        )
        return

    retrieved_context = _format_context(chunks)
    system_prompt = get_system_prompt(request.role, retrieved_context, truncated_history)
    messages = _build_messages(truncated_history, request.content)

    main_usage: LLMUsage | None = None
    async for text, usage in llm.call_main_stream(messages, system_prompt):
        if text is not None:
            yield ChatEvent(event="token", data={"text": text})
        else:
            main_usage = usage

    if main_usage is None:
        settings = get_settings()
        main_usage = LLMUsage(model=settings.main_model, prompt_tokens=0, completion_tokens=0)

    yield ChatEvent(
        event="done",
        data=ChatDoneEvent(
            citations=citations,
            is_groundable=True,
            confidence=grounding_result.confidence,
            suggested_owner_id=None,
            prompt_version=PROMPT_VERSION,
            token_usage=TokenUsage(
                main=_to_usage_detail(main_usage),
                rewrite=_to_usage_detail(rewrite_usage) if rewrite_usage else None,
                context_truncated=context_truncated,
            ),
        ).model_dump(),
    )


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


def _merge_usage(base: LLMUsage | None, extra: LLMUsage | None) -> LLMUsage | None:
    """두 LLMUsage를 합산합니다. 같은 모델(REWRITE_MODEL)을 가정합니다."""
    if extra is None:
        return base
    if base is None:
        return extra
    return LLMUsage(
        model=base.model,
        prompt_tokens=base.prompt_tokens + extra.prompt_tokens,
        completion_tokens=base.completion_tokens + extra.completion_tokens,
    )

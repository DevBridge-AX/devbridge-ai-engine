"""
신뢰도/그라운딩 판정.

두 단계로 동작합니다.
1차 필터: retriever의 최고 유사도 점수가 _LOW_SIMILARITY_THRESHOLD 미만이면 LLM 호출 없이
          is_groundable=False 즉시 반환.
2차 판단: LLM structured output({is_groundable, confidence})으로 최종 판단.

주의:
- 임계치 비교 및 알림 트리거(UC-04)는 Spring 백엔드의 책임입니다.
  이 모듈은 is_groundable/confidence 값을 산출하여 반환만 합니다.
- suggested_owner_id: is_groundable=False일 때 유사도가 가장 높은 git_commit 청크의
  author_id(USERS.id, String)를 반환합니다. 해당 청크가 없으면 None.
"""

from dataclasses import dataclass, field

from app.core.llm import provider as llm
from app.core.rag.retriever import RetrievedChunk

_LOW_SIMILARITY_THRESHOLD = 0.3

_GROUNDING_SYSTEM_PROMPT = """\
당신은 검색된 컨텍스트가 사용자 질문에 충분한 근거를 제공하는지 평가하는 전문가입니다.

평가 기준:
- is_groundable: 검색된 컨텍스트로 질문에 답변할 수 있으면 true, 그렇지 않으면 false.
- confidence: 답변 가능성에 대한 확신도 (0.0 ~ 1.0).

반드시 아래 JSON 형식으로만 응답하세요. 다른 텍스트는 포함하지 마세요.
{"is_groundable": true/false, "confidence": 0.0~1.0}\
"""


@dataclass
class GroundingResult:
    is_groundable: bool
    confidence: float
    # GIT_COMMITS.author_id (USERS.id UUID String). chat 엔드포인트에서 전달 시
    # ChatDoneEvent.suggested_owner_id(Optional[int])와 타입 불일치 주의 — 추후 정렬 필요.
    suggested_owner_id: str | None = field(default=None)


async def assess(
    retrieved_chunks: list[RetrievedChunk],
    user_query: str,
) -> GroundingResult:
    """retrieved_chunks의 유사도 및 LLM 자기평가를 결합해 그라운딩 결과를 산출합니다."""
    if not retrieved_chunks:
        return GroundingResult(is_groundable=False, confidence=0.0)

    best_score = max(c.similarity_score for c in retrieved_chunks)

    if best_score < _LOW_SIMILARITY_THRESHOLD:
        return GroundingResult(
            is_groundable=False,
            confidence=0.0,
            suggested_owner_id=_find_suggested_owner(retrieved_chunks),
        )

    context = _format_context(retrieved_chunks)
    messages = [
        {
            "role": "user",
            "content": f"질문: {user_query}\n\n검색된 컨텍스트:\n{context}",
        }
    ]

    raw = await llm.call_structured(messages, system_prompt=_GROUNDING_SYSTEM_PROMPT, max_tokens=64)

    is_groundable = bool(raw.get("is_groundable", False))
    confidence = float(raw.get("confidence", 0.0))
    suggested_owner_id = None if is_groundable else _find_suggested_owner(retrieved_chunks)

    return GroundingResult(
        is_groundable=is_groundable,
        confidence=confidence,
        suggested_owner_id=suggested_owner_id,
    )


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(f"[{i}] ({chunk.source_type}) {chunk.title}\n{chunk.content}")
    return "\n\n".join(lines)


def _find_suggested_owner(chunks: list[RetrievedChunk]) -> str | None:
    """유사도가 가장 높은 git_commit 청크의 author_id를 반환합니다."""
    git_chunks = [c for c in chunks if c.source_type == "git_commit" and c.author_id]
    if not git_chunks:
        return None
    return max(git_chunks, key=lambda c: c.similarity_score).author_id

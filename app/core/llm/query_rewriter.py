"""
멀티턴 쿼리 재구성.

turn 1 (conversation_history 길이 = 0): 재구성 없이 user_query 그대로 반환, usage=None.
turn 2+: 대화 이력을 바탕으로 맥락이 독립적인 검색 쿼리로 재구성 후 반환.

모든 LLM 호출은 provider.call_rewrite()를 통해서만 수행합니다.
"""

from app.core.llm import provider as llm
from app.core.llm.provider import LLMUsage

_SYSTEM_PROMPT = """\
당신은 한국어 기업 지식베이스를 위한 검색 쿼리 최적화 전문가입니다.
사용자의 최신 질문과 이전 대화 내용을 바탕으로, 맥락이 없어도 독립적으로 검색 가능한 단일 쿼리로 재작성하세요.

규칙:
- 이전 대화에서 생략된 주어, 대상, 맥락을 명시적으로 복원하세요.
- 만약 사용자의 최신 질문이 단순한 인사, 감사 인사, 대화형 리액션 등 검색이 필요 없는 일상적인 잡담(Chitchat)인 경우, 쿼리를 재작성하지 말고 사용자가 입력한 원래 질문 그대로 출력하세요.
- 재작성된 쿼리 텍스트만 출력하세요. 설명이나 따옴표는 포함하지 마세요.
- 원래 질문의 핵심 의도를 유지하면서 검색에 최적화된 형태로 작성하세요.
- 한국어를 유지하세요.

[Few-Shot 예시]
이전 대화:
사용자: 결제 API 문서 보여줘
어시스턴트: (문서 출력)
새 질문: "고마워!"
출력: 고마워!

이전 대화:
사용자: 결제 API 문서 보여줘
어시스턴트: (문서 출력)
새 질문: "그거 소스 코드는 어디 있어?"
출력: 결제 API 소스 코드 위치
"""

async def rewrite(
    user_query: str,
    conversation_history: list[dict],
) -> tuple[str, LLMUsage | None]:
    """쿼리를 검색에 최적화된 독립적인 쿼리로 재구성합니다.

    Args:
        user_query: 현재 사용자 입력.
        conversation_history: 이전 대화 목록 [{"role": "user"/"assistant", "content": "..."}, ...].

    Returns:
        (재구성된 쿼리 또는 원본, LLMUsage | None). turn 1이면 usage=None.
    """
    if not conversation_history:
        return user_query, None

    formatted = _format_history(conversation_history)
    messages = [
        {
            "role": "user",
            "content": (
                f"[이전 대화]\n{formatted}\n\n"
                f"[새 질문]\n{user_query}\n\n"
                "위 이전 대화를 참고하여 새 질문을 독립적인 검색 쿼리로 재작성하세요."
            ),
        }
    ]

    text, usage = await llm.call_rewrite(messages, system_prompt=_SYSTEM_PROMPT)
    return text.strip(), usage


def _format_history(history: list[dict]) -> str:
    lines = []
    for msg in history:
        role = "사용자" if msg.get("role") == "user" else "어시스턴트"
        lines.append(f"{role}: {msg.get('content', '')}")
    return "\n".join(lines)

"""
직무별 시스템 프롬프트 템플릿 (6종).

대상 역할: 기획자, 개발자, QA, 디자이너, 운영자, 신규투입자
동적 변수: retrieved_context, conversation_history, user_preference_summary

user_preference_summary는 (user_id, workspace_id) 복합키로 격리된 데이터입니다.
한 사용자가 여러 워크스페이스에 속할 수 있으므로 워크스페이스 간 데이터 혼입이
없어야 합니다.

TODO:
- PERSONA_PROMPTS: dict[PersonaRole, str] 템플릿 작성 (6종)
- render_persona_prompt(role, retrieved_context, conversation_history, user_preference_summary) 구현
"""

from enum import Enum


class PersonaRole(str, Enum):
    """직무별 페르소나 역할 6종."""

    PLANNER = "planner"  # 기획자
    DEVELOPER = "developer"  # 개발자
    QA = "qa"  # QA
    DESIGNER = "designer"  # 디자이너
    OPERATOR = "operator"  # 운영자
    NEWCOMER = "newcomer"  # 신규투입자


# TODO: 역할별 시스템 프롬프트 템플릿 작성
PERSONA_PROMPTS: dict[PersonaRole, str] = {}


def render_persona_prompt(
    role: PersonaRole,
    retrieved_context: str,
    conversation_history: list[dict[str, str]],
    user_preference_summary: str | None,
) -> str:
    """역할별 시스템 프롬프트를 동적 변수와 함께 렌더링. TODO: 구현."""
    raise NotImplementedError

"""
app/core/llm/persona_prompts.py 유닛 테스트.

외부 의존성 없는 순수 함수 테스트입니다.
"""

import pytest

from app.core.llm.persona_prompts import PersonaRole, get_system_prompt


ALL_ROLES = list(PersonaRole)

SAMPLE_CONTEXT = "## 로그인 API\nPOST /api/auth/login 엔드포인트입니다."
SAMPLE_HISTORY = [
    {"role": "user", "content": "인증 흐름이 어떻게 되나요?"},
    {"role": "assistant", "content": "JWT 기반입니다."},
]


@pytest.mark.parametrize("role", ALL_ROLES)
def test_all_roles_render_without_error(role):
    prompt = get_system_prompt(role, SAMPLE_CONTEXT, [])
    assert isinstance(prompt, str)
    assert len(prompt) > 0


@pytest.mark.parametrize("role", ALL_ROLES)
def test_retrieved_context_in_prompt(role):
    prompt = get_system_prompt(role, SAMPLE_CONTEXT, [])
    assert SAMPLE_CONTEXT in prompt


@pytest.mark.parametrize("role", ALL_ROLES)
def test_with_history_shows_section(role):
    prompt = get_system_prompt(role, SAMPLE_CONTEXT, SAMPLE_HISTORY)
    assert "[이전 대화]" in prompt


@pytest.mark.parametrize("role", ALL_ROLES)
def test_without_history_no_section(role):
    prompt = get_system_prompt(role, SAMPLE_CONTEXT, [])
    assert "[이전 대화]" not in prompt


def test_history_user_label():
    prompt = get_system_prompt(PersonaRole.DEVELOPER, SAMPLE_CONTEXT, SAMPLE_HISTORY)
    assert "사용자: 인증 흐름이 어떻게 되나요?" in prompt


def test_history_assistant_label():
    prompt = get_system_prompt(PersonaRole.DEVELOPER, SAMPLE_CONTEXT, SAMPLE_HISTORY)
    assert "어시스턴트: JWT 기반입니다." in prompt


def test_empty_context_uses_fallback():
    prompt = get_system_prompt(PersonaRole.NEWCOMER, "", [])
    assert "관련 컨텍스트를 찾지 못했습니다" in prompt


def test_persona_role_enum_values():
    assert PersonaRole.PLANNER == "planner"
    assert PersonaRole.DEVELOPER == "developer"
    assert PersonaRole.QA == "qa"
    assert PersonaRole.DESIGNER == "designer"
    assert PersonaRole.OPERATOR == "operator"
    assert PersonaRole.NEWCOMER == "newcomer"


def test_common_instructions_present():
    """모든 역할의 프롬프트에 공통 규칙 섹션이 포함되어야 합니다."""
    for role in ALL_ROLES:
        prompt = get_system_prompt(role, SAMPLE_CONTEXT, [])
        assert "규칙:" in prompt, f"규칙 섹션 없음: {role}"

"""app/core/rag/grounding_prompts.py 유닛 테스트."""

import pytest

from app.core.llm import provider
from app.core.rag.grounding_prompts import (
    GROUNDING_PROMPT_V1,
    GROUNDING_PROMPT_V2_STRICT,
    GROUNDING_PROMPTS,
    get_grounding_prompt,
)

_JSON_FORMAT = '{"is_groundable": true/false, "confidence": 0.0~1.0}'


def test_v1_equals_provider_prompt():
    assert get_grounding_prompt("v1") == provider._GROUNDING_SYSTEM_PROMPT
    assert GROUNDING_PROMPT_V1 == provider._GROUNDING_SYSTEM_PROMPT


def test_v2_strict_has_format_and_explicit_rule():
    text = get_grounding_prompt("v2-strict")
    assert _JSON_FORMAT in text
    assert "명시" in text
    assert text != GROUNDING_PROMPT_V1


def test_v1_and_v2_share_json_format_line():
    assert _JSON_FORMAT in GROUNDING_PROMPT_V1
    assert _JSON_FORMAT in GROUNDING_PROMPT_V2_STRICT


def test_registry_keys():
    assert set(GROUNDING_PROMPTS) == {"v1", "v2-strict"}


def test_unknown_version_raises_value_error_with_valid_keys():
    with pytest.raises(ValueError) as exc:
        get_grounding_prompt("v9")
    assert "v1" in str(exc.value)
    assert "v2-strict" in str(exc.value)

"""app/core/rag/grounding_prompts.py 유닛 테스트."""

import hashlib

import pytest

from app.core.llm import provider
from app.core.rag.grounding_prompts import (
    GROUNDING_PROMPT_V1,
    GROUNDING_PROMPT_V2_STRICT,
    GROUNDING_PROMPTS,
    get_grounding_prompt,
)

_JSON_FORMAT = '{"is_groundable": true/false, "confidence": 0.0~1.0}'


# 분리 이전 develop(a1f6d5e)의 provider.py `_GROUNDING_SYSTEM_PROMPT` 원문 SHA-256.
# v1 텍스트가 의도치 않게 바뀌지 않았는지 고정값으로 검증합니다(같은 값끼리 비교하면 항상 통과).
_V1_ORIGINAL_SHA256 = "1f202a7e9733a3b756d85ee2ccd3429604a36c28498f0fe3d1ce085f9b9b082d"


def test_v1_matches_original_text_hash():
    assert hashlib.sha256(GROUNDING_PROMPT_V1.encode("utf-8")).hexdigest() == _V1_ORIGINAL_SHA256
    assert hashlib.sha256(get_grounding_prompt("v1").encode("utf-8")).hexdigest() == _V1_ORIGINAL_SHA256


def test_provider_alias_points_to_v1():
    assert provider._GROUNDING_SYSTEM_PROMPT is GROUNDING_PROMPT_V1


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

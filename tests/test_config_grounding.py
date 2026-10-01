"""그라운딩 판정 설정값 검증 테스트 (잘못된 값은 기동 시 ValidationError)."""

import typing

import pytest
from pydantic import ValidationError

from app.config import Settings
from app.core.rag.grounding_prompts import GROUNDING_PROMPTS


def _settings(**kwargs) -> Settings:
    return Settings(_env_file=None, **kwargs)


def test_defaults_keep_current_behaviour():
    s = _settings()
    assert s.grounding_prompt_version == "v1"
    assert s.grounding_judge_top_k == 5
    assert s.grounding_judge_max_chunk_chars == 0


def test_valid_values_accepted():
    s = _settings(grounding_prompt_version="v2-strict", grounding_judge_top_k=3, grounding_judge_max_chunk_chars=600)
    assert s.grounding_prompt_version == "v2-strict"
    assert s.grounding_judge_top_k == 3
    assert s.grounding_judge_max_chunk_chars == 600


@pytest.mark.parametrize("version", ["v2", "V1", ""])
def test_unknown_prompt_version_rejected(version):
    with pytest.raises(ValidationError):
        _settings(grounding_prompt_version=version)


@pytest.mark.parametrize("top_k", [0, -1])
def test_top_k_must_be_positive(top_k):
    with pytest.raises(ValidationError):
        _settings(grounding_judge_top_k=top_k)


def test_max_chunk_chars_must_not_be_negative():
    with pytest.raises(ValidationError):
        _settings(grounding_judge_max_chunk_chars=-1)


def test_prompt_version_literal_matches_prompt_registry():
    annotation = Settings.model_fields["grounding_prompt_version"].annotation
    assert set(typing.get_args(annotation)) == set(GROUNDING_PROMPTS)

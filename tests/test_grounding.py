"""
app/core/rag/grounding.py 유닛 테스트.

provider.call_grounding을 모킹하여 LLM 호출 없이 검증합니다.
임계치는 config.grounding_similarity_threshold(기본값 0.35)를 사용합니다.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.config import get_settings
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult, assess
from app.core.rag.retriever import RetrievedChunk

_THRESHOLD = get_settings().grounding_similarity_threshold
_MOCK_USAGE = LLMUsage(model="gemini-3.5-flash-lite", prompt_tokens=30, completion_tokens=10)


def _make_chunk(
    chunk_id: int,
    source_type: str,
    similarity_score: float,
    author_id: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        source_type=source_type,
        source_id=chunk_id,
        title=f"chunk-{chunk_id}",
        content="내용",
        similarity_score=similarity_score,
        author_id=author_id,
    )


def _grounding_ok(is_groundable: bool, confidence: float):
    """call_grounding mock 반환값 (dict, LLMUsage)."""
    return {"is_groundable": is_groundable, "confidence": confidence}, _MOCK_USAGE


async def test_empty_chunks_returns_not_groundable():
    result = await assess([], "질문")
    assert result.is_groundable is False
    assert result.confidence == 0.0
    assert result.suggested_owner_id is None
    assert result.llm_usage is None


async def test_low_similarity_returns_not_groundable_without_llm():
    chunks = [_make_chunk(1, "document", _THRESHOLD - 0.01)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        result = await assess(chunks, "질문")
        mock_llm.assert_not_called()

    assert result.is_groundable is False
    assert result.confidence == 0.0
    assert result.llm_usage is None


async def test_high_similarity_calls_llm():
    chunks = [_make_chunk(1, "document", _THRESHOLD + 0.1)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(True, 0.9)
        result = await assess(chunks, "질문")
        mock_llm.assert_called_once()

    assert result.is_groundable is True
    assert result.confidence == pytest.approx(0.9)
    assert result.llm_usage == _MOCK_USAGE


async def test_grounding_prompt_contains_query_and_context():
    """call_grounding에 전달되는 prompt에 질문과 컨텍스트가 포함되어야 합니다."""
    chunks = [_make_chunk(1, "document", _THRESHOLD + 0.1)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(True, 0.9)
        await assess(chunks, "인증 흐름이 어떻게 되나요?")

    prompt_arg = mock_llm.call_args.args[0]
    assert "인증 흐름이 어떻게 되나요?" in prompt_arg
    assert "chunk-1" in prompt_arg  # title


async def test_groundable_true_suggested_owner_is_none():
    chunks = [_make_chunk(1, "git_commit", 0.8, author_id="uuid-abc")]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(True, 0.85)
        result = await assess(chunks, "질문")

    assert result.suggested_owner_id is None


async def test_not_groundable_returns_best_git_author():
    chunks = [
        _make_chunk(1, "git_commit", 0.5, author_id="uuid-low"),
        _make_chunk(2, "git_commit", 0.75, author_id="uuid-high"),
        _make_chunk(3, "document", 0.9, author_id=None),
    ]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(False, 0.2)
        result = await assess(chunks, "질문")

    # document(0.9)가 더 높지만 git_commit 중 0.75가 최고
    assert result.suggested_owner_id == "uuid-high"


async def test_not_groundable_no_git_chunks_owner_none():
    chunks = [_make_chunk(1, "document", 0.8)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(False, 0.1)
        result = await assess(chunks, "질문")

    assert result.suggested_owner_id is None


async def test_low_similarity_with_git_chunk_returns_author():
    """1차 필터(유사도 미달)에서도 git chunk author_id를 반환해야 합니다."""
    chunks = [_make_chunk(1, "git_commit", 0.1, author_id="uuid-owner")]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        result = await assess(chunks, "질문")
        mock_llm.assert_not_called()

    assert result.is_groundable is False
    assert result.suggested_owner_id == "uuid-owner"


async def test_llm_fallback_dict_propagates():
    """call_grounding 폴백({"is_groundable": False}) 반환값이 그대로 전달됩니다."""
    chunks = [_make_chunk(1, "document", 0.8)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = ({"is_groundable": False, "confidence": 0.0}, _MOCK_USAGE)
        result = await assess(chunks, "질문")

    assert result.is_groundable is False
    assert result.llm_usage == _MOCK_USAGE


async def test_grounding_result_llm_usage_default_none():
    result = GroundingResult(is_groundable=True, confidence=0.95)
    assert result.suggested_owner_id is None
    assert result.llm_usage is None
    assert result.fallback_reason is None


async def test_llm_exception_sets_fallback_reason_llm_error():
    """call_grounding 호출이 예외를 던지면 fallback_reason="llm_error"로 유사도 fallback 적용."""
    chunks = [_make_chunk(1, "document", _THRESHOLD + 0.1)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.side_effect = RuntimeError("gms unreachable")
        result = await assess(chunks, "질문")

    assert result.is_groundable is True
    assert result.confidence == pytest.approx(_THRESHOLD + 0.1)
    assert result.llm_usage is None
    assert result.fallback_reason == "llm_error"


async def test_parse_failure_sets_fallback_reason_parse_error():
    """call_grounding이 빈 dict({})를 반환하면 fallback_reason="parse_error"로 유사도 fallback 적용."""
    chunks = [_make_chunk(1, "document", _THRESHOLD + 0.1)]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = ({}, _MOCK_USAGE)
        result = await assess(chunks, "질문")

    assert result.is_groundable is True
    assert result.confidence == pytest.approx(_THRESHOLD + 0.1)
    assert result.llm_usage == _MOCK_USAGE
    assert result.fallback_reason == "parse_error"


# ---------------------------------------------------------------------------
# build_grounding_prompt: top_k / max_chunk_chars (판정 컨텍스트 축소)
# ---------------------------------------------------------------------------

from types import SimpleNamespace

from app.core.rag import grounding as grounding_module
from app.core.rag.grounding import build_grounding_prompt


def _long_chunks(n: int, content_len: int = 50) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=i,
            source_type="document",
            source_id=i,
            title=f"chunk-{i}",
            content="가" * content_len,
            similarity_score=0.9 - i * 0.01,
        )
        for i in range(1, n + 1)
    ]


def _patch_judge_settings(monkeypatch, top_k=5, max_chars=0):
    monkeypatch.setattr(
        grounding_module,
        "get_settings",
        lambda: SimpleNamespace(
            grounding_similarity_threshold=_THRESHOLD,
            grounding_judge_top_k=top_k,
            grounding_judge_max_chunk_chars=max_chars,
        ),
    )


def test_build_prompt_top_k_limits_chunks():
    prompt = build_grounding_prompt(_long_chunks(5), "질문", top_k=3, max_chunk_chars=0)
    assert "[1]" in prompt and "[2]" in prompt and "[3]" in prompt
    assert "[4]" not in prompt and "[5]" not in prompt
    assert "chunk-3" in prompt and "chunk-4" not in prompt


def test_build_prompt_truncates_with_marker():
    prompt = build_grounding_prompt(_long_chunks(2, 50), "질문", top_k=2, max_chunk_chars=10)
    assert "가" * 10 + "…(생략)" in prompt
    assert "가" * 11 not in prompt


def test_build_prompt_no_marker_when_content_short():
    prompt = build_grounding_prompt(_long_chunks(1, 5), "질문", top_k=1, max_chunk_chars=10)
    assert "…(생략)" not in prompt


def test_build_prompt_defaults_from_settings_keep_five_no_truncation(monkeypatch):
    _patch_judge_settings(monkeypatch)
    prompt = build_grounding_prompt(_long_chunks(7, 2000), "질문")
    assert "[5]" in prompt and "[6]" not in prompt
    assert "…(생략)" not in prompt
    assert "가" * 2000 in prompt


def test_build_prompt_settings_override(monkeypatch):
    _patch_judge_settings(monkeypatch, top_k=2, max_chars=20)
    prompt = build_grounding_prompt(_long_chunks(5, 100), "질문")
    assert "[2]" in prompt and "[3]" not in prompt
    assert "…(생략)" in prompt


async def test_assess_calls_llm_once_with_settings_based_prompt(monkeypatch):
    _patch_judge_settings(monkeypatch, top_k=3, max_chars=0)
    chunks = _long_chunks(5)

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(True, 0.9)
        result = await assess(chunks, "질문")

    mock_llm.assert_called_once()
    prompt = mock_llm.call_args.args[0]
    assert "[3]" in prompt and "[4]" not in prompt
    assert result.is_groundable is True


async def test_assess_owner_suggestion_uses_full_chunk_list(monkeypatch):
    _patch_judge_settings(monkeypatch, top_k=1, max_chars=0)
    chunks = [
        _make_chunk(1, "document", 0.9),
        _make_chunk(2, "git_commit", 0.8, author_id="u-42"),
    ]

    with patch("app.core.rag.grounding.llm.call_grounding", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _grounding_ok(False, 0.1)
        result = await assess(chunks, "질문")

    assert result.suggested_owner_id == "u-42"

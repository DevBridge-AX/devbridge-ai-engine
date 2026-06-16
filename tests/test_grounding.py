"""
app/core/rag/grounding.py 유닛 테스트.

provider.call_rewrite를 모킹하여 LLM 호출 없이 검증합니다.
임계치는 config.grounding_similarity_threshold(기본값 0.4)를 사용합니다.
"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from app.config import get_settings
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult, assess
from app.core.rag.retriever import RetrievedChunk

_THRESHOLD = get_settings().grounding_similarity_threshold
_MOCK_USAGE = LLMUsage(model="claude-haiku-4-5-20251001", prompt_tokens=30, completion_tokens=10)


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


def _rewrite_ok(is_groundable: bool, confidence: float):
    """call_rewrite mock 반환값 (json_str, LLMUsage)."""
    return json.dumps({"is_groundable": is_groundable, "confidence": confidence}), _MOCK_USAGE


async def test_empty_chunks_returns_not_groundable():
    result = await assess([], "질문")
    assert result.is_groundable is False
    assert result.confidence == 0.0
    assert result.suggested_owner_id is None
    assert result.llm_usage is None


async def test_low_similarity_returns_not_groundable_without_llm():
    chunks = [_make_chunk(1, "document", _THRESHOLD - 0.01)]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        result = await assess(chunks, "질문")
        mock_llm.assert_not_called()

    assert result.is_groundable is False
    assert result.confidence == 0.0
    assert result.llm_usage is None


async def test_high_similarity_calls_llm():
    chunks = [_make_chunk(1, "document", _THRESHOLD + 0.1)]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _rewrite_ok(True, 0.9)
        result = await assess(chunks, "질문")
        mock_llm.assert_called_once()

    assert result.is_groundable is True
    assert result.confidence == pytest.approx(0.9)
    assert result.llm_usage == _MOCK_USAGE


async def test_groundable_true_suggested_owner_is_none():
    chunks = [_make_chunk(1, "git_commit", 0.8, author_id="uuid-abc")]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _rewrite_ok(True, 0.85)
        result = await assess(chunks, "질문")

    assert result.suggested_owner_id is None


async def test_not_groundable_returns_best_git_author():
    chunks = [
        _make_chunk(1, "git_commit", 0.5, author_id="uuid-low"),
        _make_chunk(2, "git_commit", 0.75, author_id="uuid-high"),
        _make_chunk(3, "document", 0.9, author_id=None),
    ]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _rewrite_ok(False, 0.2)
        result = await assess(chunks, "질문")

    # document(0.9)가 더 높지만 git_commit 중 0.75가 최고
    assert result.suggested_owner_id == "uuid-high"


async def test_not_groundable_no_git_chunks_owner_none():
    chunks = [_make_chunk(1, "document", 0.8)]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = _rewrite_ok(False, 0.1)
        result = await assess(chunks, "질문")

    assert result.suggested_owner_id is None


async def test_low_similarity_with_git_chunk_returns_author():
    """1차 필터(유사도 미달)에서도 git chunk author_id를 반환해야 합니다."""
    chunks = [_make_chunk(1, "git_commit", 0.1, author_id="uuid-owner")]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        result = await assess(chunks, "질문")
        mock_llm.assert_not_called()

    assert result.is_groundable is False
    assert result.suggested_owner_id == "uuid-owner"


async def test_llm_json_parse_error_falls_back_to_not_groundable():
    """LLM이 잘못된 JSON을 반환하면 is_groundable=False 폴백."""
    chunks = [_make_chunk(1, "document", 0.8)]

    with patch("app.core.rag.grounding.llm.call_rewrite", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = ("not valid json", _MOCK_USAGE)
        result = await assess(chunks, "질문")

    assert result.is_groundable is False


async def test_grounding_result_llm_usage_default_none():
    result = GroundingResult(is_groundable=True, confidence=0.95)
    assert result.suggested_owner_id is None
    assert result.llm_usage is None

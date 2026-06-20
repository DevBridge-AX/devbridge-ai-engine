"""
app/core/llm/query_rewriter.py 유닛 테스트.

provider.call_rewrite를 모킹하여 LLM 호출 없이 검증합니다.
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.core.llm.provider import LLMUsage
from app.core.llm.query_rewriter import rewrite

_MOCK_USAGE = LLMUsage(model="claude-haiku-4-5-20251001", prompt_tokens=80, completion_tokens=20)


async def test_turn1_empty_history_returns_original():
    """turn 1 (빈 history)은 원본 쿼리를 그대로 반환하고 usage=None이어야 합니다."""
    query = "RAG 아키텍처가 뭔가요?"
    result_query, usage = await rewrite(query, conversation_history=[])

    assert result_query == query
    assert usage is None


async def test_turn1_does_not_call_llm():
    with patch("app.core.llm.query_rewriter.llm.call_rewrite", new_callable=AsyncMock) as mock_rewrite:
        await rewrite("질문", conversation_history=[])
        mock_rewrite.assert_not_called()


async def test_turn2_calls_llm():
    history = [{"role": "user", "content": "이전 질문"}, {"role": "assistant", "content": "이전 답변"}]
    rewritten = "재구성된 독립 쿼리"

    with patch("app.core.llm.query_rewriter.llm.call_rewrite", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = (rewritten, _MOCK_USAGE)
        result_query, usage = await rewrite("그게 뭔가요?", conversation_history=history)

    assert result_query == rewritten
    assert usage == _MOCK_USAGE
    mock_rewrite.assert_called_once()


async def test_turn2_message_contains_user_label():
    history = [{"role": "user", "content": "유저 메시지"}]

    with patch("app.core.llm.query_rewriter.llm.call_rewrite", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = ("재구성", _MOCK_USAGE)
        await rewrite("새 질문", conversation_history=history)

    messages = mock_rewrite.call_args.args[0]
    assert any("사용자: 유저 메시지" in m["content"] for m in messages)


async def test_turn2_message_contains_assistant_label():
    history = [
        {"role": "user", "content": "질문"},
        {"role": "assistant", "content": "어시스턴트 답변"},
    ]

    with patch("app.core.llm.query_rewriter.llm.call_rewrite", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = ("재구성", _MOCK_USAGE)
        await rewrite("후속 질문", conversation_history=history)

    messages = mock_rewrite.call_args.args[0]
    assert any("어시스턴트: 어시스턴트 답변" in m["content"] for m in messages)


async def test_result_is_stripped():
    history = [{"role": "user", "content": "질문"}]

    with patch("app.core.llm.query_rewriter.llm.call_rewrite", new_callable=AsyncMock) as mock_rewrite:
        mock_rewrite.return_value = ("  공백이 있는 쿼리  ", _MOCK_USAGE)
        result_query, _ = await rewrite("질문", conversation_history=history)

    assert result_query == "공백이 있는 쿼리"

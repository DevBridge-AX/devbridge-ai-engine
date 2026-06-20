"""
app/core/embeddings/embedder.py 유닛 테스트.

httpx.AsyncClient를 모킹하여 외부 API 호출 없이 검증합니다.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.core.embeddings.embedder import _TOKENS_PER_TEXT, EmbedResult, embed_texts


def _make_mock_client(embeddings: list[list[float]]):
    """지정된 임베딩 벡터를 반환하는 httpx 클라이언트 mock을 생성합니다."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "embeddings": [{"values": v} for v in embeddings]
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    return mock_client


def _patch_client(mock_client):
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


async def test_single_text_returns_correct_embedding():
    vectors = [[0.1, 0.2, 0.3]]
    mock_client = _make_mock_client(vectors)

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["RAG 아키텍처 설계 지침 문서"])

    assert result.embeddings == vectors


async def test_total_tokens_per_text():
    mock_client = _make_mock_client([[0.1, 0.2]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["텍스트 하나"])

    assert result.total_tokens == pytest.approx(1 * _TOKENS_PER_TEXT)


async def test_total_tokens_multiple_texts():
    n = 5
    mock_client = _make_mock_client([[0.1]] * n)

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["텍스트"] * n)

    assert result.total_tokens == pytest.approx(n * _TOKENS_PER_TEXT)


async def test_auth_header_uses_goog_api_key():
    mock_client = _make_mock_client([[0.0]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        await embed_texts(["test"])

    _, kwargs = mock_client.post.call_args
    headers = kwargs.get("headers") or mock_client.post.call_args[0][1] if len(mock_client.post.call_args[0]) > 1 else kwargs["headers"]
    call_kwargs = mock_client.post.call_args.kwargs
    assert "x-goog-api-key" in call_kwargs.get("headers", {})


async def test_request_url_contains_batch_embed():
    mock_client = _make_mock_client([[0.0]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        await embed_texts(["test"])

    url_arg = mock_client.post.call_args.args[0]
    assert "batchEmbedContents" in url_arg


async def test_request_body_gemini_format():
    mock_client = _make_mock_client([[0.0]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        await embed_texts(["안녕하세요"])

    body = mock_client.post.call_args.kwargs["json"]
    assert "requests" in body
    assert "content" in body["requests"][0]
    assert "parts" in body["requests"][0]["content"]
    assert body["requests"][0]["content"]["parts"][0]["text"] == "안녕하세요"


async def test_batch_over_100_splits_into_two_calls():
    """101개 텍스트는 100 + 1로 분할되어 API를 2회 호출해야 합니다."""
    n = 101
    # 첫 번째 배치(100개)와 두 번째 배치(1개)의 응답을 순서대로 반환
    resp1 = MagicMock()
    resp1.raise_for_status = MagicMock()
    resp1.json.return_value = {"embeddings": [{"values": [float(i)]} for i in range(100)]}

    resp2 = MagicMock()
    resp2.raise_for_status = MagicMock()
    resp2.json.return_value = {"embeddings": [{"values": [100.0]}]}

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[resp1, resp2])

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=ctx):
        result = await embed_texts(["t"] * n)

    assert mock_client.post.call_count == 2
    assert len(result.embeddings) == n
    assert result.total_tokens == pytest.approx(n * _TOKENS_PER_TEXT)


async def test_empty_texts_raises_value_error():
    with pytest.raises(ValueError, match="texts must not be empty"):
        await embed_texts([])


async def test_embed_result_model_fields():
    mock_client = _make_mock_client([[0.1, 0.2]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["test"])

    assert isinstance(result, EmbedResult)
    assert result.embedding_model  # 비어있지 않아야 함
    assert result.embedding_model_version  # 비어있지 않아야 함

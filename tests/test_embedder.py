"""
app/core/embeddings/embedder.py 유닛 테스트.

httpx.AsyncClient를 모킹하여 외부 API 호출 없이 검증합니다.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, call, patch

from app.config import get_settings
from app.core.embeddings.embedder import EmbedResult, embed_texts

_TOKENS_PER_TEXT = get_settings().embedding_tokens_per_text


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


async def test_total_tokens_uses_config_constant(monkeypatch):
    """total_tokens는 module 상수가 아니라 settings.embedding_tokens_per_text를 사용해야 합니다."""
    from types import SimpleNamespace

    from app.core.embeddings import embedder

    custom_settings = SimpleNamespace(
        gemini_base_url="https://gms.example.com/gemini",
        embedding_model="fake-embedding-model",
        embedding_model_version="v9",
        gms_api_key="test-key",
        embedding_tokens_per_text=1.5,  # 기본값(0.2)과 다른 값으로 오버라이드
    )
    monkeypatch.setattr(embedder, "get_settings", lambda: custom_settings)

    n = 3
    mock_client = _make_mock_client([[0.1]] * n)

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["텍스트"] * n)

    assert result.total_tokens == pytest.approx(n * 1.5)


async def test_token_source_is_estimate():
    """EmbedResult.token_source는 실측이 아닌 추정치임을 항상 나타내야 합니다."""
    mock_client = _make_mock_client([[0.1, 0.2]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["test"])

    assert result.token_source == "estimate_per_text"


async def test_provider_tokens_summed_across_batches():
    """2개 배치 각각의 usageMetadata.promptTokenCount가 합산되어야 하며,
    total_tokens(추정치)는 이 값과 무관하게 그대로 유지되어야 합니다."""
    n = 150  # 100 + 50, 배치 2회 분할
    resp1 = MagicMock()
    resp1.raise_for_status = MagicMock()
    resp1.json.return_value = {
        "embeddings": [{"values": [float(i)]} for i in range(100)],
        "usageMetadata": {"promptTokenCount": 90, "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 90}]},
    }

    resp2 = MagicMock()
    resp2.raise_for_status = MagicMock()
    resp2.json.return_value = {
        "embeddings": [{"values": [float(i)]} for i in range(50)],
        "usageMetadata": {"promptTokenCount": 45, "promptTokenDetails": [{"modality": "TEXT", "tokenCount": 45}]},
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[resp1, resp2])

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=ctx):
        result = await embed_texts(["t"] * n)

    assert mock_client.post.call_count == 2
    assert result.provider_tokens == 90 + 45
    assert result.total_tokens == pytest.approx(n * _TOKENS_PER_TEXT)


async def test_provider_tokens_none_when_usage_missing():
    """배치 중 하나라도 usageMetadata(promptTokenCount)가 없으면 provider_tokens는
    None이어야 하며, total_tokens(추정치)는 이 값과 무관하게 그대로 유지되어야 합니다."""
    n = 150
    resp1 = MagicMock()
    resp1.raise_for_status = MagicMock()
    resp1.json.return_value = {
        "embeddings": [{"values": [float(i)]} for i in range(100)],
        "usageMetadata": {"promptTokenCount": 90},
    }

    resp2 = MagicMock()
    resp2.raise_for_status = MagicMock()
    resp2.json.return_value = {
        "embeddings": [{"values": [float(i)]} for i in range(50)],
        # 두 번째 배치는 usageMetadata가 없음
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[resp1, resp2])

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_client)
    ctx.__aexit__ = AsyncMock(return_value=False)

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=ctx):
        result = await embed_texts(["t"] * n)

    assert result.provider_tokens is None
    assert result.total_tokens == pytest.approx(n * _TOKENS_PER_TEXT)


# --- taskType (선택 인자) -----------------------------------------------------


def _posted_requests(mock_client) -> list[list[dict]]:
    """post 호출별 body["requests"] 목록."""
    return [c.kwargs["json"]["requests"] for c in mock_client.post.call_args_list]


async def test_no_task_type_by_default_keeps_body():
    mock_client = _make_mock_client([[0.1]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["텍스트"])

    (requests,) = _posted_requests(mock_client)
    assert all("taskType" not in r for r in requests)
    assert set(requests[0]) == {"model", "content"}
    assert result.task_type is None


async def test_task_type_added_to_each_request():
    mock_client = _make_mock_client([[0.1], [0.2]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["a", "b"], task_type="SEMANTIC_SIMILARITY")

    (requests,) = _posted_requests(mock_client)
    assert [r["taskType"] for r in requests] == ["SEMANTIC_SIMILARITY"] * 2
    assert requests[0]["content"] == {"parts": [{"text": "a"}]}
    assert result.task_type == "SEMANTIC_SIMILARITY"


async def test_task_type_applied_to_every_batch():
    n = 150
    resp1, resp2 = MagicMock(), MagicMock()
    for resp, size in ((resp1, 100), (resp2, 50)):
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"embeddings": [{"values": [0.1]}] * size}
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=[resp1, resp2])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        result = await embed_texts(["t"] * n, task_type="RETRIEVAL_QUERY")

    batches = _posted_requests(mock_client)
    assert [len(b) for b in batches] == [100, 50]
    assert all(r["taskType"] == "RETRIEVAL_QUERY" for b in batches for r in b)
    assert result.task_type == "RETRIEVAL_QUERY"


async def test_unknown_task_type_raises_before_request():
    mock_client = _make_mock_client([[0.1]])

    with patch("app.core.embeddings.embedder.httpx.AsyncClient", return_value=_patch_client(mock_client)):
        with pytest.raises(ValueError, match="task_type"):
            await embed_texts(["a"], task_type="NOPE")

    mock_client.post.assert_not_called()


def test_query_embed_kwargs_follows_setting(monkeypatch):
    from types import SimpleNamespace

    from app.core.embeddings import embedder

    monkeypatch.setattr(embedder, "get_settings", lambda: SimpleNamespace(embedding_query_task_type=None))
    assert embedder.query_embed_kwargs() == {}

    monkeypatch.setattr(
        embedder, "get_settings", lambda: SimpleNamespace(embedding_query_task_type="SEMANTIC_SIMILARITY")
    )
    assert embedder.query_embed_kwargs() == {"task_type": "SEMANTIC_SIMILARITY"}

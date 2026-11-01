"""
app/core/rag/chunk_context.py 유닛 테스트.

call_with_model과 asyncio.sleep을 monkeypatch해 네트워크·대기 없이 검증합니다.
"""

import asyncio
from types import SimpleNamespace

import httpx
import pytest

from app.config import get_settings
from app.core.llm.provider import LLMUsage
from app.core.rag import chunk_context
from app.core.rag.chunk_context import content_sha1, generate_chunk_contexts
from app.core.rag.chunker import Chunk


def _chunks(n: int) -> list[Chunk]:
    return [Chunk(content=f"청크 본문 {i}", chunk_metadata={}) for i in range(n)]


def _settings(monkeypatch, **overrides):
    base = dict(
        chunk_context_enabled=True,
        chunk_context_model="fake-model",
        chunk_context_max_doc_chars=1000,
        chunk_context_concurrency=4,
        chunk_context_bm25=False,
        chunk_context_max_retries=3,
    )
    base.update(overrides)
    ns = SimpleNamespace(**base)
    monkeypatch.setattr(chunk_context, "get_settings", lambda: ns)
    return ns


def _run(coro):
    return asyncio.run(coro)


def _status_error(code: int) -> httpx.HTTPStatusError:
    req = httpx.Request("POST", "http://x")
    return httpx.HTTPStatusError("err", request=req, response=httpx.Response(code, request=req))


async def _yield():
    """이벤트 루프에 제어를 넘긴다(asyncio.sleep이 패치되어도 동작하도록 Future 사용)."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()
    loop.call_soon(fut.set_result, None)
    await fut


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps: list[float] = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(chunk_context.asyncio, "sleep", fake_sleep)
    return sleeps


def test_defaults_are_off():
    s = get_settings()
    assert s.chunk_context_enabled is False
    assert s.chunk_context_bm25 is False
    assert s.chunk_context_max_retries == 3


def test_disabled_makes_no_calls(monkeypatch):
    _settings(monkeypatch, chunk_context_enabled=False)
    calls = []

    async def fake(*a, **kw):
        calls.append(1)
        return "x", LLMUsage("m", 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("제목", "문서", _chunks(3)))
    assert calls == []
    assert res.contexts == [None, None, None]
    assert res.llm_calls == 0


def test_enabled_calls_per_chunk(monkeypatch):
    _settings(monkeypatch)
    seen = []

    async def fake(model, messages, system_prompt, *, max_tokens, purpose, timeout=30.0, **kw):
        seen.append((model, purpose, max_tokens, messages[0]["content"]))
        return "  문맥입니다  ", LLMUsage(model, 10, 5)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("배포 가이드", "문서 전문", _chunks(3)))
    assert res.contexts == ["문맥입니다"] * 3
    assert res.llm_calls == 3
    assert res.failed == 0 and res.reused == 0 and res.retried == 0
    assert res.usage_prompt_tokens == 30 and res.usage_completion_tokens == 15
    assert all(m == "fake-model" and p == "chunk_context" and t == 120 for m, p, t, _ in seen)
    assert "문서 제목: 배포 가이드" in seen[0][3]
    assert "<chunk>" in seen[0][3]


def test_concurrency_limit(monkeypatch):
    _settings(monkeypatch, chunk_context_concurrency=2)
    state = {"cur": 0, "max": 0}

    async def fake(model, *a, **kw):
        state["cur"] += 1
        state["max"] = max(state["max"], state["cur"])
        await _yield()
        state["cur"] -= 1
        return "ctx", LLMUsage(model, 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(6)))
    assert state["max"] == 2
    assert res.llm_calls == 6


def test_failed_chunks_are_none(monkeypatch):
    _settings(monkeypatch)

    async def fake(model, messages, *a, **kw):
        body = messages[0]["content"]
        if "청크 본문 1" in body:
            raise RuntimeError("boom")
        if "청크 본문 2" in body:
            return "   ", LLMUsage(model, 1, 0)
        return "ok", LLMUsage(model, 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(4)))
    assert res.contexts == ["ok", None, None, "ok"]
    assert res.failed == 2


def test_reuse_skips_llm(monkeypatch):
    _settings(monkeypatch)
    calls = []

    async def fake(model, messages, *a, **kw):
        calls.append(messages[0]["content"])
        return "new", LLMUsage(model, 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    chunks = _chunks(3)
    reuse = {content_sha1(chunks[0].content): "old0", content_sha1(chunks[2].content): "old2"}
    res = _run(generate_chunk_contexts("t", "d", chunks, reuse=reuse))
    assert res.contexts == ["old0", "new", "old2"]
    assert res.reused == 2
    assert res.llm_calls == 1
    assert len(calls) == 1


def test_context_truncated_to_300(monkeypatch):
    _settings(monkeypatch)

    async def fake(model, *a, **kw):
        return "가" * 500, LLMUsage(model, 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(1)))
    assert len(res.contexts[0]) == 300


def test_doc_truncated(monkeypatch):
    _settings(monkeypatch, chunk_context_max_doc_chars=1000)
    bodies = []

    async def fake(model, messages, *a, **kw):
        bodies.append(messages[0]["content"])
        return "c", LLMUsage(model, 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    _run(generate_chunk_contexts("t", "A" * 5000, _chunks(1)))
    assert "A" * 1000 + "…(생략)" in bodies[0]
    assert "A" * 1001 not in bodies[0]


def test_retry_on_429_then_success(monkeypatch, no_sleep):
    _settings(monkeypatch)
    attempts = {"n": 0}

    async def fake(model, *a, **kw):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise _status_error(429)
        return "ok", LLMUsage(model, 1, 1)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(1)))
    assert res.contexts == ["ok"]
    assert res.retried == 2
    assert res.llm_calls == 3
    assert res.failed == 0
    assert no_sleep == [15, 30]


def test_retry_exhausted_is_failed(monkeypatch, no_sleep):
    _settings(monkeypatch, chunk_context_max_retries=3)

    async def fake(*a, **kw):
        raise _status_error(429)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(1)))
    assert res.contexts == [None]
    assert res.failed == 1
    assert res.retried == 3
    assert no_sleep == [15, 30, 60]


def test_non_429_not_retried(monkeypatch, no_sleep):
    _settings(monkeypatch)

    async def fake(*a, **kw):
        raise _status_error(500)

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(1)))
    assert res.failed == 1 and res.retried == 0 and no_sleep == []


def test_timeout_not_retried(monkeypatch, no_sleep):
    _settings(monkeypatch)

    async def fake(*a, **kw):
        raise httpx.ReadTimeout("slow")

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    res = _run(generate_chunk_contexts("t", "d", _chunks(1)))
    assert res.failed == 1 and res.retried == 0 and no_sleep == []


def test_cancelled_error_propagates(monkeypatch):
    _settings(monkeypatch)

    async def fake(*a, **kw):
        raise asyncio.CancelledError()

    monkeypatch.setattr(chunk_context, "call_with_model", fake)
    with pytest.raises(asyncio.CancelledError):
        _run(generate_chunk_contexts("t", "d", _chunks(2)))

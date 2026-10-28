"""
document_ingestion 청크 문맥 보강 연결 테스트.

off일 때 임베딩 입력·chunk_metadata가 현행과 같은지, on일 때 임베딩 입력이 "context\\n\\ncontent"이고
DocumentChunk.content는 원문 그대로인지, 재인덱싱 시 content_sha1이 같은 청크는 LLM을 부르지 않는지,
메트릭 필드가 기록되는지 검증합니다. 네트워크 호출 없음(call_with_model·embed_texts 모두 fake).
"""

import asyncio
import json

import httpx
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.core import metrics
from app.core.llm.provider import LLMUsage
from app.core.rag import chunk_context
from app.core.rag.chunk_context import content_sha1
from app.core.rag.chunker import chunk_document
from app.db.models import DocumentChunk, KnowledgeDocument
from app.pipelines import document_ingestion

DOC_TEXT = "# 제목\n\n첫 번째 단락입니다.\n\n## 소제목\n\n두 번째 단락입니다."


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    DocumentChunk.__table__.create(engine)
    KnowledgeDocument.__table__.create(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture(autouse=True)
def metrics_tmp_dir(monkeypatch, tmp_path):
    metrics_dir = tmp_path / "metrics"
    monkeypatch.setattr(
        metrics, "get_settings", lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(metrics_dir))
    )
    return metrics_dir


def _read_records(metrics_dir) -> list[dict]:
    path = metrics_dir / "ingestion.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


@pytest.fixture
def env(monkeypatch, tmp_path):
    """fake embed/store + fake call_with_model. embed 입력과 LLM 호출을 기록해 돌려준다."""
    rec = SimpleNamespace(embed_inputs=[], llm_calls=[])

    async def fake_embed(texts):
        rec.embed_inputs.append(list(texts))
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
            provider_tokens=len(texts) * 3,
        )

    async def fake_llm(model, messages, system_prompt, **kw):
        rec.llm_calls.append(messages[0]["content"])
        return "문맥 문장", LLMUsage(model, 5, 3)

    monkeypatch.setattr(document_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(
        document_ingestion, "get_vector_store", lambda: SimpleNamespace(add=lambda **kw: None)
    )
    monkeypatch.setattr(
        document_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=lambda workspace_id: None)
    )
    monkeypatch.setattr(document_ingestion, "log_embedding_usage", lambda *a, **kw: None)
    monkeypatch.setattr(chunk_context, "call_with_model", fake_llm)

    doc = tmp_path / "doc.md"
    doc.write_text(DOC_TEXT, encoding="utf-8")
    rec.path = str(doc)
    return rec


def _set_enabled(monkeypatch, enabled: bool):
    base = get_settings()
    patched = base.model_copy(update={"chunk_context_enabled": enabled, "chunk_context_model": "fake-model"})
    monkeypatch.setattr(document_ingestion, "get_settings", lambda: patched)
    monkeypatch.setattr(chunk_context, "get_settings", lambda: patched)


def _run(db, env):
    asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", env.path, "markdown"))
    db.flush()


def test_disabled_matches_current_behavior(db, env, monkeypatch, metrics_tmp_dir):
    _set_enabled(monkeypatch, False)
    expected = chunk_document(DOC_TEXT, "markdown")

    _run(db, env)

    assert env.llm_calls == []
    assert env.embed_inputs == [[c.content for c in expected]]
    rows = db.execute(select(DocumentChunk).order_by(DocumentChunk.id)).scalars().all()
    assert [r.content for r in rows] == [c.content for c in expected]
    assert [r.chunk_metadata for r in rows] == [c.chunk_metadata for c in expected]
    record = _read_records(metrics_tmp_dir)[0]
    assert record["embedding_input_chars"] == sum(len(c.content) for c in expected)
    assert record["context_llm_calls"] == 0
    assert record["context_failed"] == 0
    assert record["context_reused"] == 0
    assert record["context_retried"] == 0
    assert record["context_ms"] is None


def test_enabled_embeds_context_and_keeps_content_unchanged(db, env, monkeypatch, metrics_tmp_dir):
    _set_enabled(monkeypatch, True)
    expected = chunk_document(DOC_TEXT, "markdown")

    _run(db, env)

    assert len(env.llm_calls) == len(expected)
    assert env.embed_inputs == [[f"문맥 문장\n\n{c.content}" for c in expected]]
    rows = db.execute(select(DocumentChunk).order_by(DocumentChunk.id)).scalars().all()
    assert [r.content for r in rows] == [c.content for c in expected]  # content_unchanged
    for row, chunk in zip(rows, expected):
        assert row.chunk_metadata["context"] == "문맥 문장"
        assert row.chunk_metadata["context_model"] == "fake-model"
        assert row.chunk_metadata["content_sha1"] == content_sha1(chunk.content)
        # 기존 메타데이터 키는 그대로 유지된다.
        assert all(row.chunk_metadata[k] == v for k, v in chunk.chunk_metadata.items())

    record = _read_records(metrics_tmp_dir)[0]
    assert record["context_llm_calls"] == len(expected)
    assert record["context_failed"] == 0
    assert record["context_reused"] == 0
    assert record["context_ms"] is not None and record["context_ms"] >= 0
    assert record["embedding_input_chars"] == sum(len(f"문맥 문장\n\n{c.content}") for c in expected)


def test_doc_title_is_passed_to_prompt(db, env, monkeypatch):
    _set_enabled(monkeypatch, True)
    db.add(KnowledgeDocument(id="doc-1", workspace_id="ws-1", source_id="s", title="배포 가이드"))
    db.flush()

    _run(db, env)

    assert env.llm_calls
    assert all("문서 제목: 배포 가이드\n" in p for p in env.llm_calls)


def test_missing_document_row_uses_empty_title(db, env, monkeypatch):
    _set_enabled(monkeypatch, True)

    _run(db, env)

    assert all("문서 제목: \n" in p for p in env.llm_calls)


def test_failed_context_falls_back_to_content_only(db, env, monkeypatch, metrics_tmp_dir):
    _set_enabled(monkeypatch, True)

    async def failing(*a, **kw):
        raise RuntimeError("llm down")

    monkeypatch.setattr(chunk_context, "call_with_model", failing)
    expected = chunk_document(DOC_TEXT, "markdown")

    _run(db, env)

    assert env.embed_inputs == [[c.content for c in expected]]
    rows = db.execute(select(DocumentChunk)).scalars().all()
    assert all(r.chunk_metadata["context"] is None and r.chunk_metadata["context_model"] is None for r in rows)
    record = _read_records(metrics_tmp_dir)[0]
    assert record["result"] == "COMPLETED"
    assert record["context_failed"] == len(expected)


def test_reindex_reuses_context_by_sha1(db, env, monkeypatch, metrics_tmp_dir):
    _set_enabled(monkeypatch, True)
    expected = chunk_document(DOC_TEXT, "markdown")

    _run(db, env)
    assert len(env.llm_calls) == len(expected)

    env.llm_calls.clear()
    env.embed_inputs.clear()
    _run(db, env)

    assert env.llm_calls == []  # sha1이 같은 청크는 LLM 미호출
    assert env.embed_inputs == [[f"문맥 문장\n\n{c.content}" for c in expected]]
    record = _read_records(metrics_tmp_dir)[-1]
    assert record["context_llm_calls"] == 0
    assert record["context_reused"] == len(expected)
    rows = db.execute(select(DocumentChunk)).scalars().all()
    assert len(rows) == len(expected)
    assert all(r.chunk_metadata["context"] == "문맥 문장" for r in rows)


def test_429_retry_is_recorded_in_metric(db, env, monkeypatch, metrics_tmp_dir):
    _set_enabled(monkeypatch, True)
    expected = chunk_document(DOC_TEXT, "markdown")
    state = {"first": True}

    async def flaky(model, messages, system_prompt, **kw):
        # 첫 호출만 429를 한 번 받고 이후는 성공한다.
        if state["first"]:
            state["first"] = False
            req = httpx.Request("POST", "http://x")
            raise httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))
        return "문맥 문장", LLMUsage(model, 5, 3)

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(chunk_context, "call_with_model", flaky)
    monkeypatch.setattr(chunk_context.asyncio, "sleep", no_sleep)

    _run(db, env)

    record = _read_records(metrics_tmp_dir)[0]
    assert record["context_retried"] == 1
    assert record["context_failed"] == 0
    assert record["context_llm_calls"] == len(expected) + 1

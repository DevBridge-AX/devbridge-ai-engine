"""
인덱싱 시점 접근 제어 스냅샷 유닛 테스트 (docs/access-control.md §3.3).

문서 인덱싱이 task_id/sensitivity_level을 document_chunks에 복사하는지,
요청 스키마 기본값이 하위호환("제한 없음")을 유지하는지 검증합니다.
임베딩 API·Chroma·BM25는 mock으로 대체하고, DB는 인메모리 SQLite를 사용합니다.
"""

import asyncio
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.core import metrics
from app.db.models import DocumentChunk
from app.pipelines import document_ingestion
from app.schemas.ingestion import (
    DocumentIngestionRequest,
    GitIngestionRequest,
    RetryIngestionRequest,
)


@pytest.fixture
def db(tmp_path):
    engine = create_engine("sqlite://")
    DocumentChunk.__table__.create(engine)
    with Session(engine) as session:
        yield session


@pytest.fixture
def patched_pipeline(monkeypatch, tmp_path):
    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    added: list[dict] = []
    monkeypatch.setattr(document_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(
        document_ingestion,
        "get_vector_store",
        lambda: SimpleNamespace(add=lambda **kw: added.append(kw)),
    )
    monkeypatch.setattr(
        document_ingestion,
        "get_bm25_manager",
        lambda: SimpleNamespace(invalidate=lambda workspace_id: None),
    )
    monkeypatch.setattr(document_ingestion, "log_embedding_usage", lambda *a, **kw: None)
    # 이 테스트는 접근 제어 스냅샷만 검증하므로 metrics_dir를 tmp_path로 돌려 레포에 파일이 남지 않게 한다.
    monkeypatch.setattr(
        metrics, "get_settings", lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(tmp_path / "metrics"))
    )

    doc = tmp_path / "doc.md"
    doc.write_text("# 제목\n\n첫 번째 단락입니다.\n\n## 소제목\n\n두 번째 단락입니다.", encoding="utf-8")
    return str(doc), added


class TestDocumentChunkSnapshot:

    def test_task_and_sensitivity_copied_to_chunks(self, db, patched_pipeline):
        file_path, _ = patched_pipeline
        asyncio.run(
            document_ingestion._run(
                db,
                "ws-1",
                "doc-1",
                file_path,
                "markdown",
                task_id="task-9",
                sensitivity_level="restricted",
            )
        )

        chunks = db.execute(select(DocumentChunk)).scalars().all()
        assert chunks
        assert all(c.task_id == "task-9" for c in chunks)
        assert all(c.sensitivity_level == "restricted" for c in chunks)

    def test_defaults_are_unrestricted(self, db, patched_pipeline):
        file_path, _ = patched_pipeline
        asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", file_path, "markdown"))

        chunks = db.execute(select(DocumentChunk)).scalars().all()
        assert chunks
        assert all(c.task_id is None for c in chunks)
        assert all(c.sensitivity_level == "normal" for c in chunks)


class TestIngestionRequestSchema:

    def test_sensitivity_defaults_to_normal(self):
        doc = DocumentIngestionRequest(
            workspace_id="ws", title="t", doc_type="markdown", file_path="/x"
        )
        git = GitIngestionRequest(workspace_id="ws", commits=[])
        retry = RetryIngestionRequest(document_id="d")
        assert doc.sensitivity_level == "normal"
        assert git.sensitivity_level == "normal"
        assert retry.sensitivity_level == "normal"

    def test_unknown_sensitivity_rejected(self):
        with pytest.raises(ValidationError):
            DocumentIngestionRequest(
                workspace_id="ws",
                title="t",
                doc_type="markdown",
                file_path="/x",
                sensitivity_level="secret",
            )

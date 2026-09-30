"""
document_ingestion 관측성 계측(C-1) 유닛 테스트.

성공/0청크(EMPTY)/단계별 실패(read/chunk/embed/store) 각각에서 ingestion.jsonl에
1건씩 기록되는지, result/failure_stage/error_type/구간별 ms 필드가 기대대로
채워지는지 검증합니다. metrics_dir는 tmp_path로 돌립니다. 문서 원문은 기록하지
않으므로 payload에 content가 없는지도 함께 확인합니다.
"""

import json
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core import metrics
from app.db.models import DocumentChunk
from app.pipelines import document_ingestion


@pytest.fixture
def db():
    engine = create_engine("sqlite://")
    DocumentChunk.__table__.create(engine)
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
def patched_success(monkeypatch, tmp_path):
    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    monkeypatch.setattr(document_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(
        document_ingestion, "get_vector_store", lambda: SimpleNamespace(add=lambda **kw: None)
    )
    monkeypatch.setattr(
        document_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=lambda workspace_id: None)
    )
    monkeypatch.setattr(document_ingestion, "log_embedding_usage", lambda *a, **kw: None)

    doc = tmp_path / "doc.md"
    doc.write_text("# 제목\n\n첫 번째 단락입니다.\n\n## 소제목\n\n두 번째 단락입니다.", encoding="utf-8")
    return str(doc)


class TestSuccessPath:

    def test_records_completed_with_all_stage_ms(self, db, patched_success, metrics_tmp_dir):
        import asyncio

        asyncio.run(
            document_ingestion._run(db, "ws-1", "doc-1", patched_success, "markdown")
        )

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]

        assert record["result"] == "COMPLETED"
        assert record["failure_stage"] is None
        assert record["error_type"] is None
        assert record["chunk_count"] > 0
        assert record["char_count"] > 0
        assert record["file_bytes"] > 0
        assert record["embedding_input_chars"] > 0
        for key in ("read_ms", "chunk_ms", "embed_ms", "store_ms", "total_ms"):
            assert record[key] is not None
            assert record[key] >= 0

        raw = json.dumps(record, ensure_ascii=False)
        assert "제목" not in raw
        assert "단락" not in raw


class TestEmptyPath:

    def test_records_empty_when_no_chunks(self, db, monkeypatch, tmp_path, metrics_tmp_dir):
        doc = tmp_path / "empty.md"
        doc.write_text("   \n\n   ", encoding="utf-8")  # 공백만 있는 파일 → chunk_document가 빈 리스트 반환

        import asyncio

        asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", str(doc), "markdown"))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]
        assert record["result"] == "EMPTY"
        assert record["chunk_count"] == 0
        assert record["failure_stage"] is None
        assert record["embedding_input_chars"] is None


class TestFailurePaths:

    def test_read_failure_records_failure_stage_read(self, db, metrics_tmp_dir):
        import asyncio

        with pytest.raises(FileNotFoundError):
            asyncio.run(
                document_ingestion._run(db, "ws-1", "doc-1", "/no/such/file.md", "markdown")
            )

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]
        assert record["result"] == "FAILED"
        assert record["failure_stage"] == "read"
        assert record["error_type"] == "FileNotFoundError"
        assert record["chunk_count"] == 0
        assert record["embedding_input_chars"] is None

    def test_embed_failure_records_failure_stage_embed(self, db, monkeypatch, tmp_path, metrics_tmp_dir):
        async def failing_embed(texts):
            raise RuntimeError("embedding api down")

        monkeypatch.setattr(document_ingestion, "embed_texts", failing_embed)

        doc = tmp_path / "doc.md"
        doc.write_text("# 제목\n\n본문 내용입니다.", encoding="utf-8")

        import asyncio

        with pytest.raises(RuntimeError):
            asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", str(doc), "markdown"))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]
        assert record["result"] == "FAILED"
        assert record["failure_stage"] == "embed"
        assert record["error_type"] == "RuntimeError"
        assert record["chunk_count"] > 0
        assert record["embedding_input_chars"] > 0

    def test_store_failure_records_failure_stage_store(self, db, monkeypatch, tmp_path, metrics_tmp_dir):
        async def fake_embed(texts):
            return SimpleNamespace(
                embeddings=[[0.1, 0.2] for _ in texts],
                embedding_model="fake",
                embedding_model_version="v0",
                total_tokens=len(texts),
            )

        def failing_vector_store():
            def add(**kw):
                raise RuntimeError("vector store down")

            return SimpleNamespace(add=add)

        monkeypatch.setattr(document_ingestion, "embed_texts", fake_embed)
        monkeypatch.setattr(document_ingestion, "get_vector_store", failing_vector_store)

        doc = tmp_path / "doc.md"
        doc.write_text("# 제목\n\n본문 내용입니다.", encoding="utf-8")

        import asyncio

        with pytest.raises(RuntimeError):
            asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", str(doc), "markdown"))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]
        assert record["result"] == "FAILED"
        assert record["failure_stage"] == "store"
        assert record["error_type"] == "RuntimeError"
        assert record["embedding_input_chars"] > 0

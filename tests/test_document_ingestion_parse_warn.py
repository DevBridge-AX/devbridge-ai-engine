"""
document_ingestion 파싱 판정 보강(C-2) 유닛 테스트.

read_text(errors="replace")는 깨진 바이너리도 치환 문자(�)로 대체해 예외 없이
"성공" 처리한다. replacement_ratio가 config.ingestion_parse_warn_ratio(기본 5%)를
초과하면 메트릭 result가 PARSE_WARN으로 구분되고, 그래도 인덱싱 자체는 정상 진행되는지
(청크가 생성되고 analysis_status 로직에 영향이 없는지) 검증합니다. 정상 UTF-8 파일은
COMPLETED/replacement_ratio=0.0으로 유지되는지도 함께 확인합니다.
"""

import asyncio
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


@pytest.fixture(autouse=True)
def patched_pipeline(monkeypatch):
    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    monkeypatch.setattr(document_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(document_ingestion, "get_vector_store", lambda: SimpleNamespace(add=lambda **kw: None))
    monkeypatch.setattr(
        document_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=lambda workspace_id: None)
    )
    monkeypatch.setattr(document_ingestion, "log_embedding_usage", lambda *a, **kw: None)


def _read_records(metrics_dir) -> list[dict]:
    path = metrics_dir / "ingestion.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class TestParseWarnDetection:

    def test_binary_file_above_threshold_is_parse_warn(self, db, tmp_path, metrics_tmp_dir):
        # 유효한 UTF-8 시작 바이트가 이어지지 않는 바이트를 섞어 치환 문자 비율이 5%를 넘게 만든다.
        garbled = b"\xff\xfe" * 40 + "정상 텍스트 일부입니다.".encode("utf-8")
        doc = tmp_path / "binary.bin"
        doc.write_bytes(garbled)

        asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", str(doc), "text"))

        record = _read_records(metrics_tmp_dir)[0]
        assert record["result"] == "PARSE_WARN"
        assert record["replacement_ratio"] > 0.05
        assert record["chunk_count"] > 0  # 인덱싱은 그대로 진행

    def test_clean_utf8_file_is_completed_with_zero_ratio(self, db, tmp_path, metrics_tmp_dir):
        doc = tmp_path / "clean.md"
        doc.write_text("# 제목\n\n정상적인 한국어 문서 본문입니다.", encoding="utf-8")

        asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", str(doc), "markdown"))

        record = _read_records(metrics_tmp_dir)[0]
        assert record["result"] == "COMPLETED"
        assert record["replacement_ratio"] == 0.0

    def test_ratio_at_or_below_threshold_stays_completed(self, db, tmp_path, metrics_tmp_dir):
        # 치환 문자 1개를 충분히 긴 정상 텍스트 안에 섞어 비율을 5% 미만으로 유지한다.
        text = "정상 문단입니다. " * 40
        doc = tmp_path / "mild.md"
        doc.write_text(text, encoding="utf-8")
        with doc.open("ab") as f:
            f.write(b"\xff")  # 단일 깨진 바이트 추가

        asyncio.run(document_ingestion._run(db, "ws-1", "doc-1", str(doc), "markdown"))

        record = _read_records(metrics_tmp_dir)[0]
        assert record["replacement_ratio"] < 0.05
        assert record["result"] == "COMPLETED"

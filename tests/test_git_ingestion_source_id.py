"""
git_ingestion source_id 처리 테스트.

GIT_COMMITS.source_id는 Spring 스키마와 맞춰 NOT NULL이다(06-23 정합성 교정).
따라서 (1) /ingestion/git 엔드포인트는 source_id/data_source_id가 모두 없으면 422로 거절하고,
(2) 파이프라인에 None이 들어오면 FAILED 메트릭으로 기록되며(조용히 성공하지 않음),
(3) 이미 저장된 source_id는 None 재호출로 덮어써지지 않아야 한다.
Spring 사용자조회(httpx)·임베딩·벡터스토어·메트릭은 mock, DB는 인메모리 SQLite입니다.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.core import metrics
from app.db.models import Base, DocumentChunk, GitCommit
from app.pipelines import git_ingestion
from app.schemas.ingestion import CommitData


def _commit(hash_suffix: str) -> CommitData:
    return CommitData(
        commit_hash=f"hash-{hash_suffix}",
        short_hash=hash_suffix[:7],
        author_name="tester",
        author_email="tester@example.com",
        message="fix: sample commit",
        committed_at=datetime.now(timezone.utc),
        branch_name="main",
    )


@pytest.fixture
def sqlite_session_local(monkeypatch):
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(git_ingestion, "SessionLocal", session_local)
    return session_local


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

    async def fake_lookup_author_id(email, settings):
        return None

    monkeypatch.setattr(git_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(git_ingestion, "get_vector_store", lambda: SimpleNamespace(add=lambda **kw: None))
    monkeypatch.setattr(
        git_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=lambda workspace_id: None)
    )
    monkeypatch.setattr(git_ingestion, "log_embedding_usage", lambda *a, **kw: None)
    monkeypatch.setattr(git_ingestion, "_lookup_author_id", fake_lookup_author_id)
    monkeypatch.setattr(
        git_ingestion,
        "get_settings",
        lambda: SimpleNamespace(
            embedding_model="fake",
            ai_analysis_mode="fallback",
            gms_api_key="",
            spring_backend_base_url="http://backend",
            spring_user_lookup_path="/lookup",
            internal_api_key="key",
        ),
    )




def _read_records(metrics_dir) -> list[dict]:
    path = metrics_dir / "git_ingestion.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _get_commit(session_local, commit_hash: str) -> GitCommit | None:
    with session_local() as db:
        return db.execute(
            select(GitCommit).where(GitCommit.commit_hash == commit_hash)
        ).scalar_one_or_none()


class TestNullSourceIdIsRejected:

    def test_pipeline_records_failed_not_silent_success(self, sqlite_session_local, metrics_tmp_dir):
        """None이 파이프라인까지 오면 NOT NULL 제약으로 FAILED가 기록되어야 한다(불변 조건 문서화)."""
        commit = _commit("aaa1111")

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", None, [commit]))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        assert records[0]["result"] == "FAILED", records[0]
        assert records[0]["error_type"] == "IntegrityError"
        assert _get_commit(sqlite_session_local, commit.commit_hash) is None

    def test_git_endpoint_returns_422_without_source_id(self):
        """엔드포인트는 백그라운드로 넘기기 전에 거절해야 한다."""
        from fastapi.testclient import TestClient
        from app.config import get_settings
        from app.main import app

        client = TestClient(app)
        headers = {"X-Internal-Api-Key": get_settings().internal_api_key}
        body = {
            "workspace_id": "ws-1",
            "commits": [_commit("ddd4444").model_dump(mode="json")],
        }
        resp = client.post("/api/ingestion/git", json=body, headers=headers)
        assert resp.status_code == 422
        assert "source_id" in resp.json()["detail"]


class TestBackfillSourceId:

    def test_existing_source_id_not_overwritten_by_none(self, sqlite_session_local, metrics_tmp_dir):
        commit = _commit("ccc3333")

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-real", [commit]))
        asyncio.run(git_ingestion.ingest_git_commits("ws-1", None, [commit]))

        row = _get_commit(sqlite_session_local, commit.commit_hash)
        assert row.source_id == "src-real"

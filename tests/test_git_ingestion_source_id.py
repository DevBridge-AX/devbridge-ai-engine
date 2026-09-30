"""
git_ingestion source_id 처리 테스트.

ingest_git_commits(source_id=None) 경로와, 이후 실제 source_id로 재호출 시
_refresh_existing_commit()이 GIT_COMMITS.source_id를 backfill하는지 검증합니다.
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


# 알려진 결함: app/db/models.py GitCommit.source_id가 nullable=False라서 source_id=None 인덱싱은
# IntegrityError(NOT NULL)로 FAILED 처리된다. 모델을 nullable=True로 고치면 strict xfail이 깨져 알려준다.
_NULL_SOURCE_ID_BUG = pytest.mark.xfail(
    strict=True,
    reason="GitCommit.source_id가 nullable=False라 source_id=None 삽입이 NOT NULL 제약으로 실패",
)


def _read_records(metrics_dir) -> list[dict]:
    path = metrics_dir / "git_ingestion.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _get_commit(session_local, commit_hash: str) -> GitCommit | None:
    with session_local() as db:
        return db.execute(
            select(GitCommit).where(GitCommit.commit_hash == commit_hash)
        ).scalar_one_or_none()


class TestIngestWithNullSourceId:

    @_NULL_SOURCE_ID_BUG
    def test_completes_and_row_has_null_source_id(self, sqlite_session_local, metrics_tmp_dir):
        commit = _commit("aaa1111")

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", None, [commit]))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        assert records[0]["result"] == "COMPLETED", records[0]

        row = _get_commit(sqlite_session_local, commit.commit_hash)
        assert row is not None
        assert row.source_id is None

        with sqlite_session_local() as db:
            chunks = db.execute(select(DocumentChunk)).scalars().all()
        assert len(chunks) > 0


class TestBackfillSourceId:

    @_NULL_SOURCE_ID_BUG
    def test_later_call_with_real_source_id_backfills(self, sqlite_session_local, metrics_tmp_dir):
        commit = _commit("bbb2222")

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", None, [commit]))
        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-real", [commit]))

        records = _read_records(metrics_tmp_dir)
        assert [r["result"] for r in records] == ["COMPLETED", "COMPLETED"], records

        row = _get_commit(sqlite_session_local, commit.commit_hash)
        assert row.source_id == "src-real"

    def test_existing_source_id_not_overwritten_by_none(self, sqlite_session_local, metrics_tmp_dir):
        commit = _commit("ccc3333")

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-real", [commit]))
        asyncio.run(git_ingestion.ingest_git_commits("ws-1", None, [commit]))

        row = _get_commit(sqlite_session_local, commit.commit_hash)
        assert row.source_id == "src-real"

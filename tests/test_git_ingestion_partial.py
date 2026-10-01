"""
git_ingestion 커밋 단위 부분 성공(WP2) 유닛 테스트.

커밋 1건 = SAVEPOINT 1개(GIT_COMMITS 생성부터 분석 upsert까지). 실패한 커밋은
GIT_COMMITS/document_chunks/분석 행이 모두 남지 않고, 그 커밋이 벡터스토어에 쓴 벡터는
삭제되어야 합니다. 메트릭은 COMPLETED / PARTIAL / FAILED로 구분됩니다.
SQLite에서 SAVEPOINT를 쓰기 위해 pysqlite 표준 우회(connect/begin 이벤트)를 적용합니다.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.core import metrics
from app.db.models import Base, DocumentChunk, GitCommit
from app.pipelines import git_ingestion
from app.schemas.ingestion import CommitData

FAIL_HASH = "hash-bbb2222"


def _commit(suffix: str) -> CommitData:
    return CommitData(
        commit_hash=f"hash-{suffix}",
        short_hash=suffix[:7],
        author_name="tester",
        author_email="tester@example.com",
        message="fix: sample commit",
        committed_at=datetime.now(timezone.utc),
        branch_name="main",
    )


@pytest.fixture
def sqlite_session_local():
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _do_connect(dbapi_connection, _record):
        dbapi_connection.isolation_level = None  # pysqlite의 자체 BEGIN 제어 해제

    @event.listens_for(engine, "begin")
    def _do_begin(conn):
        conn.exec_driver_sql("BEGIN")

    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def metrics_tmp_dir(monkeypatch, tmp_path):
    metrics_dir = tmp_path / "metrics"
    monkeypatch.setattr(
        metrics, "get_settings", lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(metrics_dir))
    )
    return metrics_dir


class FakeVectorStore:
    def __init__(self):
        self.fail_hashes: set[str] = set()
        self.added: list[str] = []
        self.deleted: list[tuple[str, str]] = []

    def add(self, vector_id, chunk_id, embedding, metadata, workspace_id):
        self.added.append(vector_id)
        if metadata["commit_hash"] in self.fail_hashes:
            raise RuntimeError("vector store down")

    def delete(self, vector_id, workspace_id):
        self.deleted.append((vector_id, workspace_id))


@pytest.fixture
def env(monkeypatch, sqlite_session_local):
    invalidated: list[str] = []
    store = FakeVectorStore()

    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    async def fake_lookup_author_id(email, settings):
        return None

    monkeypatch.setattr(git_ingestion, "SessionLocal", sqlite_session_local)
    monkeypatch.setattr(git_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(git_ingestion, "get_vector_store", lambda: store)
    monkeypatch.setattr(
        git_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=invalidated.append)
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
    return SimpleNamespace(store=store, invalidated=invalidated, session_local=sqlite_session_local)


def _records(metrics_dir) -> list[dict]:
    path = metrics_dir / "git_ingestion.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _persisted_hashes(session_local) -> set[str]:
    with session_local() as db:
        return set(db.execute(select(GitCommit.commit_hash)).scalars().all())


def _chunk_count(session_local) -> int:
    with session_local() as db:
        return len(db.execute(select(DocumentChunk)).scalars().all())


class TestPartialSuccess:

    def test_middle_commit_vector_failure_is_isolated(self, env, metrics_tmp_dir):
        env.store.fail_hashes = {FAIL_HASH}
        commits = [_commit("aaa1111"), _commit("bbb2222"), _commit("ccc3333")]

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", commits))

        # SAVEPOINT 경계가 GIT_COMMITS 생성까지 포함하므로 실패 커밋은 행 자체가 없다.
        assert _persisted_hashes(env.session_local) == {"hash-aaa1111", "hash-ccc3333"}
        with env.session_local() as db:
            chunk_commit_ids = {c.source_id for c in db.execute(select(DocumentChunk)).scalars().all()}
            ok_ids = set(db.execute(select(GitCommit.id)).scalars().all())
        assert chunk_commit_ids == ok_ids

        # 실패 커밋이 add를 시도한 벡터 id만 삭제되었고, 성공 커밋 벡터는 삭제되지 않았다.
        assert len(env.store.added) == 3
        deleted_ids = [vid for vid, _ in env.store.deleted]
        assert deleted_ids == [env.store.added[1]]
        assert env.store.deleted[0][1] == "ws-1"

        records = _records(metrics_tmp_dir)
        assert len(records) == 1
        assert records[0]["result"] == "PARTIAL"
        assert records[0]["failed_commit_count"] == 1
        assert records[0]["commit_count"] == 3
        assert records[0]["error_type"] == "RuntimeError"
        assert FAIL_HASH not in json.dumps(records[0])
        assert env.invalidated == ["ws-1"]

    def test_embedding_failure_leaves_no_rows_and_nothing_to_delete(self, env, monkeypatch, metrics_tmp_dir):
        calls = {"n": 0}

        async def flaky_embed(texts):
            calls["n"] += 1
            if calls["n"] == 2:
                raise ValueError("embedding api down")
            return SimpleNamespace(
                embeddings=[[0.1, 0.2] for _ in texts],
                embedding_model="fake",
                embedding_model_version="v0",
                total_tokens=len(texts),
            )

        monkeypatch.setattr(git_ingestion, "embed_texts", flaky_embed)
        commits = [_commit("aaa1111"), _commit("bbb2222"), _commit("ccc3333")]

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", commits))

        assert _persisted_hashes(env.session_local) == {"hash-aaa1111", "hash-ccc3333"}
        assert env.store.deleted == []  # 임베딩 단계 실패는 벡터를 쓰기 전이다

        record = _records(metrics_tmp_dir)[0]
        assert record["result"] == "PARTIAL"
        assert record["failed_commit_count"] == 1
        assert record["error_type"] == "ValueError"

    def test_cleanup_failure_is_best_effort(self, env, metrics_tmp_dir):
        env.store.fail_hashes = {FAIL_HASH}

        def broken_delete(vector_id, workspace_id):
            raise OSError("chroma unavailable")

        env.store.delete = broken_delete

        asyncio.run(
            git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("aaa1111"), _commit("bbb2222")])
        )

        assert _persisted_hashes(env.session_local) == {"hash-aaa1111"}
        assert _records(metrics_tmp_dir)[0]["result"] == "PARTIAL"

    def test_all_commits_fail_records_failed(self, env, monkeypatch, metrics_tmp_dir):
        async def failing_embed(texts):
            raise RuntimeError("embedding api down")

        monkeypatch.setattr(git_ingestion, "embed_texts", failing_embed)

        asyncio.run(
            git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("aaa1111"), _commit("bbb2222")])
        )

        assert _persisted_hashes(env.session_local) == set()
        assert _chunk_count(env.session_local) == 0
        assert env.invalidated == []  # 성공 커밋이 없으면 BM25 무효화 생략

        records = _records(metrics_tmp_dir)
        assert len(records) == 1
        assert records[0]["result"] == "FAILED"
        assert records[0]["failed_commit_count"] == 2
        assert records[0]["error_type"] == "RuntimeError"

    def test_all_success_records_completed_with_zero_failed(self, env, metrics_tmp_dir):
        asyncio.run(
            git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("aaa1111"), _commit("bbb2222")])
        )

        record = _records(metrics_tmp_dir)[0]
        assert record["result"] == "COMPLETED"
        assert record["failed_commit_count"] == 0
        assert record["error_type"] is None
        assert env.invalidated == ["ws-1"]


class TestSemanticCacheInvalidation:

    def test_success_invalidates_semantic_cache(self, env, monkeypatch, metrics_tmp_dir):
        invalidated: list[str] = []
        monkeypatch.setattr(
            git_ingestion,
            "get_semantic_cache",
            lambda: SimpleNamespace(invalidate_workspace=invalidated.append),
        )

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("aaa1111")]))

        assert invalidated == ["ws-1"]

    def test_all_failed_skips_invalidation(self, env, monkeypatch, metrics_tmp_dir):
        invalidated: list[str] = []
        monkeypatch.setattr(
            git_ingestion,
            "get_semantic_cache",
            lambda: SimpleNamespace(invalidate_workspace=invalidated.append),
        )

        async def failing_embed(texts):
            raise RuntimeError("embedding api down")

        monkeypatch.setattr(git_ingestion, "embed_texts", failing_embed)

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("aaa1111")]))

        assert invalidated == []

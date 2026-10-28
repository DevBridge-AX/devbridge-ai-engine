"""
git_ingestion 커밋 분석 LLM 호출 병렬화(COMMIT_ANALYSIS_CONCURRENCY) 유닛 테스트.

_run은 [1단계 인덱싱(순차) -> 2단계 분석(병렬) -> 3단계 upsert(순차)]로 동작한다.
네트워크 호출 없이 _llm_analyze_commit을 mock으로 대체하고, SQLite 메모리 DB를 사용한다.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.core import metrics
from app.db.models import Base, DocumentChunk, GitCommit, GitCommitAnalysis
from app.pipelines import git_ingestion
from app.schemas.ingestion import CommitData

LLM_SUMMARY = "LLM 분석 결과"


def _commits(n: int) -> list[CommitData]:
    return [
        CommitData(
            commit_hash=f"hash-{i:07d}",
            short_hash=f"{i:07d}",
            author_name="tester",
            author_email="tester@example.com",
            message=f"fix: sample commit {i}",
            committed_at=datetime.now(timezone.utc),
            branch_name="main",
        )
        for i in range(n)
    ]


@pytest.fixture
def sqlite_session_local():
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _do_connect(dbapi_connection, _record):
        dbapi_connection.isolation_level = None

    @event.listens_for(engine, "begin")
    def _do_begin(conn):
        conn.exec_driver_sql("BEGIN")

    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture
def metrics_dir(monkeypatch, tmp_path):
    d = tmp_path / "metrics"
    monkeypatch.setattr(
        metrics, "get_settings", lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(d))
    )
    return d


@pytest.fixture
def env(monkeypatch, sqlite_session_local, metrics_dir):
    state = SimpleNamespace(
        llm_calls=[], in_flight=0, max_in_flight=0, settings={}, session_local=sqlite_session_local
    )

    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    async def fake_llm_analyze(commit, commit_text):
        state.llm_calls.append(commit.commit_hash)
        state.in_flight += 1
        state.max_in_flight = max(state.max_in_flight, state.in_flight)
        try:
            await asyncio.sleep(0.01)
        finally:
            state.in_flight -= 1
        return {
            "summary": LLM_SUMMARY,
            "impact_area": "backend",
            "risk_level": "LOW",
            "next_action": "확인",
        }

    async def fake_lookup_author_id(email, settings):
        return None

    def make_settings():
        base = dict(
            embedding_model="fake",
            ai_analysis_mode="llm",
            gms_api_key="key",
            spring_backend_base_url="http://backend",
            spring_user_lookup_path="/lookup",
            internal_api_key="key",
        )
        base.update(state.settings)
        return SimpleNamespace(**base)

    monkeypatch.setattr(git_ingestion, "SessionLocal", sqlite_session_local)
    monkeypatch.setattr(git_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(
        git_ingestion, "get_vector_store", lambda: SimpleNamespace(add=lambda **kw: None, delete=lambda **kw: None)
    )
    monkeypatch.setattr(
        git_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=lambda workspace_id: None)
    )
    monkeypatch.setattr(
        git_ingestion, "get_semantic_cache", lambda: SimpleNamespace(invalidate_workspace=lambda ws: None)
    )
    monkeypatch.setattr(git_ingestion, "log_embedding_usage", lambda *a, **kw: None)
    monkeypatch.setattr(git_ingestion, "_lookup_author_id", fake_lookup_author_id)
    monkeypatch.setattr(git_ingestion, "_llm_analyze_commit", fake_llm_analyze)
    monkeypatch.setattr(git_ingestion, "get_settings", make_settings)
    return state


def _ingest(commits):
    asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", commits))


def _metric(metrics_dir) -> dict:
    lines = (metrics_dir / "git_ingestion.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line][-1]


def _analysis_hashes(session_local) -> set[str]:
    with session_local() as db:
        rows = db.execute(
            select(GitCommit.commit_hash).join(GitCommitAnalysis, GitCommitAnalysis.commit_id == GitCommit.id)
        ).scalars().all()
        return set(rows)


def test_concurrency_1_calls_in_index_order(env, metrics_dir):
    env.settings["commit_analysis_concurrency"] = 1
    commits = _commits(5)

    _ingest(commits)

    assert env.llm_calls == [c.commit_hash for c in commits]
    assert env.max_in_flight == 1
    assert _analysis_hashes(env.session_local) == {c.commit_hash for c in commits}
    rec = _metric(metrics_dir)
    assert rec["result"] == "COMPLETED"
    assert rec["analysis_llm_count"] == 5
    assert rec["analysis_capped_count"] == 0


def test_max_in_flight_respects_concurrency(env, metrics_dir):
    env.settings["commit_analysis_concurrency"] = 3
    commits = _commits(6)

    _ingest(commits)

    assert 2 <= env.max_in_flight <= 3
    assert len(env.llm_calls) == 6
    assert _analysis_hashes(env.session_local) == {c.commit_hash for c in commits}
    assert _metric(metrics_dir)["result"] == "COMPLETED"


def test_cap_order_with_concurrency(env, metrics_dir):
    env.settings.update(commit_analysis_concurrency=3, commit_analysis_max_per_batch=2)
    commits = _commits(6)

    _ingest(commits)

    assert sorted(env.llm_calls) == [commits[0].commit_hash, commits[1].commit_hash]
    assert len(_analysis_hashes(env.session_local)) == 6
    rec = _metric(metrics_dir)
    assert rec["analysis_llm_count"] == 2
    assert rec["analysis_capped_count"] == 4


def test_stage2_exception_fails_only_that_commit(env, metrics_dir, monkeypatch):
    env.settings.update(commit_analysis_concurrency=2, ai_analysis_mode="fallback")
    commits = _commits(4)
    bad = commits[2].commit_hash
    orig = git_ingestion._fallback_analyze_commit

    def flaky_fallback(commit, commit_text):
        if commit.commit_hash == bad:
            raise ValueError("fallback broken")
        return orig(commit, commit_text)

    monkeypatch.setattr(git_ingestion, "_fallback_analyze_commit", flaky_fallback)

    _ingest(commits)

    assert _analysis_hashes(env.session_local) == {c.commit_hash for c in commits} - {bad}
    rec = _metric(metrics_dir)
    assert rec["result"] == "PARTIAL"
    assert rec["failed_commit_count"] == 1
    assert rec["error_type"] == "ValueError"


def test_stage3_upsert_fail_keeps_commit_and_chunks(env, metrics_dir, monkeypatch):
    commits = _commits(3)
    bad = commits[1].commit_hash
    orig = git_ingestion._upsert_commit_analysis

    def flaky_upsert(db, workspace_id, git_commit, analysis, vector_id):
        if git_commit.commit_hash == bad:
            raise RuntimeError("upsert down")
        return orig(
            db=db, workspace_id=workspace_id, git_commit=git_commit, analysis=analysis, vector_id=vector_id
        )

    monkeypatch.setattr(git_ingestion, "_upsert_commit_analysis", flaky_upsert)

    _ingest(commits)

    with env.session_local() as db:
        assert {c.commit_hash for c in db.scalars(select(GitCommit)).all()} == {c.commit_hash for c in commits}
        bad_commit = db.scalars(select(GitCommit).where(GitCommit.commit_hash == bad)).one()
        assert db.scalars(select(DocumentChunk).where(DocumentChunk.source_id == bad_commit.id)).all()
    assert _analysis_hashes(env.session_local) == {c.commit_hash for c in commits} - {bad}
    rec = _metric(metrics_dir)
    assert rec["result"] == "PARTIAL"
    assert rec["failed_commit_count"] == 1
    assert rec["error_type"] == "RuntimeError"


def test_stage1_failure_still_isolated(env, metrics_dir, monkeypatch):
    env.settings["commit_analysis_concurrency"] = 3
    commits = _commits(4)
    orig = git_ingestion._index_commit_if_needed

    async def flaky(**kw):
        if kw["commit"].commit_hash == commits[1].commit_hash:
            raise RuntimeError("index down")
        return await orig(**kw)

    monkeypatch.setattr(git_ingestion, "_index_commit_if_needed", flaky)

    _ingest(commits)

    assert commits[1].commit_hash not in env.llm_calls
    assert _analysis_hashes(env.session_local) == {c.commit_hash for c in commits} - {commits[1].commit_hash}
    rec = _metric(metrics_dir)
    assert rec["result"] == "PARTIAL"
    assert rec["failed_commit_count"] == 1


def test_metric_has_analysis_ms_and_concurrency(env, metrics_dir):
    env.settings["commit_analysis_concurrency"] = 2

    _ingest(_commits(3))

    rec = _metric(metrics_dir)
    assert isinstance(rec["analysis_ms"], float)
    assert rec["analysis_concurrency"] == 2


def test_config_concurrency_default_and_validation():
    assert Settings(_env_file=None).commit_analysis_concurrency == 1
    assert Settings(_env_file=None, commit_analysis_concurrency=4).commit_analysis_concurrency == 4
    with pytest.raises(ValidationError):
        Settings(_env_file=None, commit_analysis_concurrency=0)

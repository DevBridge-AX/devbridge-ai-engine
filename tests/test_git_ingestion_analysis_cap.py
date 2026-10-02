"""
git_ingestion 커밋 분석 LLM 호출 상한(COMMIT_ANALYSIS_MAX_PER_BATCH) 유닛 테스트.

push 1회(배치)당 LLM 분석을 시도할 최대 커밋 수를 제한하고, 초과분은 LLM 호출 없이 fallback
휴리스틱으로 분석합니다. 상한은 분석에만 영향을 주며 모든 커밋의 인덱싱/분석 upsert는 그대로
수행됩니다. 네트워크 호출 없이 _llm_analyze_commit을 mock으로 대체합니다.
"""

import asyncio
import json
import logging
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
    state = SimpleNamespace(llm_calls=[], fail_hashes=set(), settings={})

    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    async def fake_llm_analyze(commit, commit_text):
        state.llm_calls.append(commit.commit_hash)
        if commit.commit_hash in state.fail_hashes:
            raise RuntimeError("llm down")
        return {
            "summary": LLM_SUMMARY,
            "impact_area": "backend",
            "risk_level": "low",
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
    state.session_local = sqlite_session_local
    return state


def _ingest(commits):
    asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", commits))


def _metric(metrics_dir) -> dict:
    lines = (metrics_dir / "git_ingestion.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line][-1]


def _summaries(session_local) -> list[str]:
    with session_local() as db:
        return [a.summary for a in db.scalars(select(GitCommitAnalysis)).all()]


def test_default_cap_zero_is_unlimited(env, metrics_dir):
    _ingest(_commits(5))

    assert len(env.llm_calls) == 5
    assert _summaries(env.session_local).count(LLM_SUMMARY) == 5
    rec = _metric(metrics_dir)
    assert rec["analysis_llm_count"] == 5
    assert rec["analysis_capped_count"] == 0


def test_cap_limits_llm_calls_and_rest_use_fallback(env, metrics_dir, caplog):
    env.settings["commit_analysis_max_per_batch"] = 2
    commits = _commits(5)

    with caplog.at_level(logging.INFO, logger=git_ingestion.logger.name):
        _ingest(commits)

    # 배치 순서상 앞의 2개만 LLM 대상
    assert env.llm_calls == [commits[0].commit_hash, commits[1].commit_hash]
    summaries = _summaries(env.session_local)
    assert len(summaries) == 5  # 모든 커밋 분석 upsert
    assert summaries.count(LLM_SUMMARY) == 2
    with env.session_local() as db:
        assert len(db.scalars(select(GitCommit)).all()) == 5
        assert len(db.scalars(select(DocumentChunk)).all()) >= 5  # 인덱싱은 전부 수행

    rec = _metric(metrics_dir)
    assert rec["analysis_llm_count"] == 2
    assert rec["analysis_capped_count"] == 3
    assert rec["commit_count"] == 5

    cap_logs = [r for r in caplog.records if "cap reached" in r.getMessage()]
    assert len(cap_logs) == 1  # 배치당 1회
    msg = cap_logs[0].getMessage()
    assert "ws-1" in msg and "cap=2" in msg and "commit_total=5" in msg
    assert "sample commit" not in msg


def test_cap_irrelevant_in_fallback_mode(env, metrics_dir):
    env.settings.update(commit_analysis_max_per_batch=2, ai_analysis_mode="fallback")

    _ingest(_commits(5))

    assert env.llm_calls == []
    assert len(_summaries(env.session_local)) == 5
    rec = _metric(metrics_dir)
    assert rec["analysis_llm_count"] == 0
    assert rec["analysis_capped_count"] == 0


def test_cap_irrelevant_without_api_key(env, metrics_dir):
    env.settings.update(commit_analysis_max_per_batch=2, gms_api_key="")

    _ingest(_commits(3))

    assert env.llm_calls == []
    assert _metric(metrics_dir)["analysis_capped_count"] == 0


def test_failed_llm_attempt_counts_toward_cap(env, metrics_dir):
    env.settings["commit_analysis_max_per_batch"] = 2
    commits = _commits(4)
    env.fail_hashes.add(commits[0].commit_hash)

    _ingest(commits)

    # 첫 커밋 실패(fallback 처리)도 시도 1회로 센다 -> LLM 호출은 총 2회
    assert env.llm_calls == [commits[0].commit_hash, commits[1].commit_hash]
    summaries = _summaries(env.session_local)
    assert len(summaries) == 4
    assert summaries.count(LLM_SUMMARY) == 1
    rec = _metric(metrics_dir)
    assert rec["analysis_llm_count"] == 2
    assert rec["analysis_capped_count"] == 2


def test_cap_not_exceeded_logs_nothing(env, caplog):
    env.settings["commit_analysis_max_per_batch"] = 10

    with caplog.at_level(logging.INFO, logger=git_ingestion.logger.name):
        _ingest(_commits(3))

    assert len(env.llm_calls) == 3
    assert not [r for r in caplog.records if "cap reached" in r.getMessage()]


def test_config_default_and_validation():
    assert Settings(_env_file=None).commit_analysis_max_per_batch == 0
    assert Settings(_env_file=None, commit_analysis_max_per_batch=7).commit_analysis_max_per_batch == 7
    with pytest.raises(ValidationError):
        Settings(_env_file=None, commit_analysis_max_per_batch=-1)

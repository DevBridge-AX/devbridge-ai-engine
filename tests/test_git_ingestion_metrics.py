"""
git_ingestion 관측성 계측(C-3) 유닛 테스트.

커밋 수, 신규/스킵 청크 수, embed_ms, total_ms, 실패 여부가 git_ingestion.jsonl에
기록되는지 검증합니다. Spring 사용자조회(httpx)·LLM 분석·임베딩·벡터스토어는 mock,
DB는 인메모리 SQLite(SingletonThreadPool로 동일 스레드 내 커넥션 간 데이터 유지)를
사용합니다. 커밋 메시지/diff 원문은 payload에 포함되지 않아야 합니다.
"""

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core import metrics
from app.db.models import Base
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
def sqlite_session_local():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


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
            provider_tokens=len(texts) * 3,
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


class TestRunAccumulatesChunkCounts:

    def test_new_then_skipped_on_reindex(self, sqlite_session_local):
        commit_a = _commit("aaa1111")

        with sqlite_session_local() as db:
            _, _, new_count, skipped_count, input_chars, provider_tokens, embed_ms, _, _ = asyncio.run(
                git_ingestion._run(db, "ws-1", "src-1", [commit_a])
            )
            db.commit()

        assert new_count > 0
        assert skipped_count == 0
        assert input_chars > 0
        assert provider_tokens is not None and provider_tokens > 0
        assert embed_ms >= 0

        with sqlite_session_local() as db:
            _, _, new_count2, skipped_count2, input_chars2, provider_tokens2, embed_ms2, _, _ = asyncio.run(
                git_ingestion._run(db, "ws-1", "src-1", [commit_a])
            )

        assert new_count2 == 0
        assert skipped_count2 > 0
        assert input_chars2 == 0
        assert provider_tokens2 == 0
        assert embed_ms2 == 0.0


class TestIngestGitCommitsMetric:

    def test_success_records_completed(self, monkeypatch, sqlite_session_local, metrics_tmp_dir):
        monkeypatch.setattr(git_ingestion, "SessionLocal", sqlite_session_local)

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("bbb2222")]))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]

        assert record["result"] == "COMPLETED"
        assert record["commit_count"] == 1
        assert record["new_chunk_count"] > 0
        assert record["skipped_chunk_count"] == 0
        assert record["embedding_input_chars"] > 0
        assert record["embedding_provider_tokens"] == record["new_chunk_count"] * 3
        assert record["embed_ms"] >= 0
        assert record["total_ms"] >= 0
        assert record["error_type"] is None

        raw = json.dumps(record, ensure_ascii=False)
        assert "fix: sample commit" not in raw

    def test_failure_records_failed(self, monkeypatch, sqlite_session_local, metrics_tmp_dir):
        monkeypatch.setattr(git_ingestion, "SessionLocal", sqlite_session_local)

        async def failing_embed(texts):
            raise RuntimeError("embedding api down")

        monkeypatch.setattr(git_ingestion, "embed_texts", failing_embed)

        asyncio.run(git_ingestion.ingest_git_commits("ws-1", "src-1", [_commit("ccc3333")]))

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]

        assert record["result"] == "FAILED"
        assert record["failed_commit_count"] == 1
        assert record["error_type"] == "RuntimeError"
        assert record["new_chunk_count"] == 0  # 전 커밋 실패: 집계값은 0(외부 예외일 때만 None)
        assert record["embedding_input_chars"] == 0  # 위와 동일한 이유로 0
        assert record["commit_count"] == 1

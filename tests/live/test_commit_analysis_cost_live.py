"""
커밋 분석 llm 모드 비용 실측(L5) 라이브 검증 (실 GMS API 호출, 과금 발생).

RUN_LIVE_LLM=1 python3 -m pytest -m live -q tests/live/test_commit_analysis_cost_live.py -s로
실행합니다(docs/ai-api-usage.md "커밋 분석 비용 실측(L5) 방법" 참고). 이 모듈 안에서만
AI_ANALYSIS_MODE=llm, COMMIT_ANALYSIS_MAX_PER_BATCH=_CAP으로 임시 전환하고
(get_settings.cache_clear() 포함) 종료 시 원래 값으로 복원합니다.

실제 배치 경로(`git_ingestion.ingest_git_commits`)를 한 번 실행합니다: 합성 커밋
_COMMIT_COUNT건(소/중/대 diff 3종 버킷)을 임베딩·인덱싱하고, 상한 _CAP건만 LLM 분석
(purpose="commit_analysis"), 나머지는 fallback으로 분석합니다. 품질은 판정하지 않고
불변식(호출 수 == min(커밋 수, 상한), 배치 메트릭, 분석 행 존재/risk_level 허용값)만
assert하며, 건당 토큰·지연 집계를 `live_run_recorder["commit_analysis_cost"]`와 stdout에
JSON으로 남깁니다(상한 N 선정 근거).

호출 예산: 메인 분석 모델 최대 _CAP회 + 커밋 _COMMIT_COUNT건 임베딩.
"""

import hashlib
import json
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.config import get_settings
from app.db.models import Base, GitCommit, GitCommitAnalysis
from app.pipelines import git_ingestion
from app.schemas.ingestion import CommitData, GitChangedFileData
from scripts.metrics_report import _percentile, load_records

pytestmark = pytest.mark.live

_CAP = 10
_COMMIT_COUNT = 12
_WORKSPACE_ID = "live-commit-cost-ws"
_ALLOWED_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}
_BUCKETS = ("small", "medium", "large")


@pytest.fixture(scope="module", autouse=True)
def commit_analysis_cap_env(live_env):
    """이 모듈 동안만 AI_ANALYSIS_MODE=llm, 상한=_CAP으로 전환하고 종료 후 원복합니다."""
    keys = {"AI_ANALYSIS_MODE": "llm", "COMMIT_ANALYSIS_MAX_PER_BATCH": str(_CAP)}
    originals = {key: os.environ.get(key) for key in keys}
    os.environ.update(keys)
    get_settings.cache_clear()

    yield

    for key, original in originals.items():
        if original is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = original
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# 합성 커밋 (결정적 생성 — 난수 미사용)
# ---------------------------------------------------------------------------

# (파일 수, 파일당 patch 줄 수) — small ≈ 5줄, medium ≈ 30~60줄, large ≈ 150~300줄
_BUCKET_SHAPE = {"small": (1, 5), "medium": (3, 14), "large": (5, 45)}

_MESSAGES = {
    "small": [
        "fix: 로그인 실패 시 에러 메시지 오타 수정",
        "chore: bump retry timeout to 5s",
        "docs: README 배포 절차 링크 갱신",
        "fix: null check for empty cart",
    ],
    "medium": [
        "feat: 결제 취소 API에 환불 사유 필드 추가",
        "refactor: extract token validation into helper",
        "feat: 알림 설정 화면에 이메일 수신 토글 추가",
        "fix: handle duplicate webhook delivery",
    ],
    "large": [
        "feat: 워크스페이스 초대 흐름 전면 개편 (만료/재발송 포함)",
        "refactor: split order service into command and query modules",
        "feat: 파일 업로드 청크 재시도 및 진행률 표시 추가",
        "perf: batch DB writes in nightly sync job",
    ],
}

_FILE_PATHS = [
    "src/main/java/com/devbridge/payment/PaymentService.java",
    "src/main/java/com/devbridge/auth/TokenValidator.java",
    "src/pages/Settings.tsx",
    "src/components/InviteDialog.tsx",
    "app/services/sync_job.py",
    "src/main/resources/application.yml",
]


def _patch(seed: int, lines: int) -> str:
    """seed에 따라 결정적으로 생성되는 unified-diff 스타일 patch."""
    body = []
    for n in range(lines):
        sign = "+" if n % 3 else "-"
        body.append(
            f"{sign}    result_{seed}_{n} = handle_step({seed}, {n}, retries={n % 4}) "
            f"# 단계 {n} 처리 및 검증"
        )
    return "@@ -1,%d +1,%d @@\n" % (lines, lines) + "\n".join(body)


def _bucket_commit(index: int) -> tuple[str, CommitData]:
    # 버킷을 순환(small, medium, large, ...)시켜 앞 _CAP건에 세 버킷이 모두 포함되고
    # 상한으로 잘리는 마지막 2건도 섞이게 합니다.
    bucket = _BUCKETS[index % len(_BUCKETS)]
    file_count, lines = _BUCKET_SHAPE[bucket]
    messages = _MESSAGES[bucket]
    commit_hash = hashlib.sha1(f"l5-commit-cost-{index}".encode()).hexdigest()
    changed_files = []
    for f in range(file_count):
        path = _FILE_PATHS[(index + f) % len(_FILE_PATHS)]
        changed_files.append(
            GitChangedFileData(
                file_path=path,
                change_type="MODIFIED",
                additions=lines - lines // 3,
                deletions=lines // 3,
                diff_summary=f"{path} 변경 ({lines}줄)",
                patch=_patch(index * 10 + f, lines),
            )
        )
    commit = CommitData(
        commit_hash=commit_hash,
        short_hash=commit_hash[:7],
        author_name="라이브검증봇",
        author_email="live-bot@example.com",
        message=messages[(index // len(_BUCKETS)) % len(messages)],
        committed_at=datetime(2026, 10, 1, 9, index, tzinfo=timezone.utc),
        branch_name="main",
        changed_files=changed_files,
    )
    return bucket, commit


def _dist(values: list[float], digits: int = 1) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "n": len(values),
        "mean": round(statistics.fmean(values), digits),
        "p50": round(_percentile(values, 50), digits),
        "p95": round(_percentile(values, 95), digits),
    }


def _aggregate(rows: list[dict]) -> dict:
    ok = [r for r in rows if not r["error_type"]]
    return {
        "calls": len(rows),
        "ok_calls": len(ok),
        "prompt_tokens": _dist([r["prompt_tokens"] for r in ok]),
        "completion_tokens": _dist([r["completion_tokens"] for r in ok]),
        "latency_ms": _dist([r["latency_ms"] for r in ok if r["latency_ms"] is not None]),
        "tokens_per_char": _dist([r["tokens_per_char"] for r in ok if r["tokens_per_char"]], digits=4),
    }


def _read_commit_analysis_calls(metrics_dir: Path) -> list[dict]:
    records = load_records(Path(metrics_dir)).get("llm_calls", [])
    return [r for r in records if r.get("purpose") == "commit_analysis"]


async def test_commit_analysis_cost_with_cap(live_env, live_run_recorder, tmp_path, monkeypatch):
    settings = get_settings()
    assert settings.ai_analysis_mode == "llm"
    assert settings.commit_analysis_max_per_batch == _CAP

    metrics_dir = Path(live_env["metrics_dir"])
    expected_llm_calls = min(_COMMIT_COUNT, _CAP)

    built = [_bucket_commit(i) for i in range(_COMMIT_COUNT)]
    buckets = [b for b, _ in built]
    commits = [c for _, c in built]
    assert set(buckets[:expected_llm_calls]) == set(_BUCKETS)

    # AI 소유 테이블만 있는 tmp sqlite DB. 실 DB(SessionLocal)는 건드리지 않습니다.
    engine = create_engine(f"sqlite:///{tmp_path}/commit_cost.db")
    Base.metadata.create_all(engine)
    session_local = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(git_ingestion, "SessionLocal", session_local)

    # 작성자 조회는 Spring 내부 API를 호출하므로 측정 대상이 아니라 막아 둡니다.
    async def _no_author(email, settings):
        return None

    monkeypatch.setattr(git_ingestion, "_lookup_author_id", _no_author)

    calls_before = len(_read_commit_analysis_calls(metrics_dir))
    await git_ingestion.ingest_git_commits(_WORKSPACE_ID, "live-commit-cost-src", commits)
    all_calls = _read_commit_analysis_calls(metrics_dir)
    new_calls = all_calls[calls_before:]

    # 배치 메트릭 (마지막 git_ingestion 레코드)
    batch_records = load_records(metrics_dir).get("git_ingestion", [])
    assert batch_records, "git_ingestion 메트릭이 기록되지 않았습니다"
    batch = batch_records[-1]

    # 불변식: LLM 호출 1건/시도(실패한 시도도 provider가 finally에서 기록하므로 센다)
    assert len(new_calls) == expected_llm_calls
    assert batch["analysis_llm_count"] == expected_llm_calls
    assert batch["analysis_capped_count"] == _COMMIT_COUNT - expected_llm_calls

    with session_local() as db:
        stored = {
            c.commit_hash: c for c in db.scalars(select(GitCommit)).all()
        }
        analyses = {a.commit_id: a for a in db.scalars(select(GitCommitAnalysis)).all()}
    for commit in commits:
        git_commit = stored.get(commit.commit_hash)
        assert git_commit is not None, f"커밋 행 없음: {commit.commit_hash}"
        analysis = analyses.get(git_commit.id)
        assert analysis is not None, f"분석 행 없음: {commit.commit_hash}"
        assert (analysis.risk_level or "").upper() in _ALLOWED_RISK_LEVELS

    # 가정: provider의 llm_calls 레코드에는 커밋 식별자가 없으므로, 배치가 커밋을 순차 처리하고
    # 커밋당 LLM 시도가 정확히 1회(레코드 1건)라는 점에 의존해 레코드 i를 커밋 i에 대응시킵니다.
    rows = []
    for i, call in enumerate(new_calls):
        commit_text = git_ingestion._build_commit_text(commits[i])
        chars = len(commit_text)
        # LLM 프롬프트의 diff 미리보기는 MAX_ANALYSIS_CHARS에서 잘리므로, tokens_per_char는
        # _llm_analyze_commit과 같은 방식으로 실제 전송한 미리보기 길이 기준으로 계산합니다
        # (시스템 프롬프트 토큰이 포함되어 짧은 커밋일수록 비율이 높게 나옵니다).
        sent_chars = len(
            git_ingestion._strip_author_email_line(commit_text)[: git_ingestion.MAX_ANALYSIS_CHARS]
        )
        prompt_tokens = int(call.get("prompt_tokens") or 0)
        rows.append(
            {
                "index": i,
                "bucket": buckets[i],
                "commit_text_chars": chars,
                "sent_chars": sent_chars,
                "diff_truncated": chars > git_ingestion.MAX_ANALYSIS_CHARS,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": int(call.get("completion_tokens") or 0),
                "latency_ms": call.get("latency_ms"),
                "error_type": call.get("error_type"),
                "tokens_per_char": round(prompt_tokens / sent_chars, 4) if sent_chars else None,
            }
        )

    ok_rows = [r for r in rows if not r["error_type"]]
    mean_total_tokens = (
        statistics.fmean([r["prompt_tokens"] + r["completion_tokens"] for r in ok_rows])
        if ok_rows
        else 0.0
    )
    summary = {
        "model": settings.main_model,
        "cap": _CAP,
        "commit_count": _COMMIT_COUNT,
        "llm_calls": batch["analysis_llm_count"],
        "capped": batch["analysis_capped_count"],
        "batch_total_ms": batch.get("total_ms"),
        "batch_embed_ms": batch.get("embed_ms"),
        "per_commit": rows,
        "aggregate_overall": _aggregate(rows),
        "aggregate_by_bucket": {
            bucket: _aggregate([r for r in rows if r["bucket"] == bucket]) for bucket in _BUCKETS
        },
        "mean_total_tokens_per_llm_commit": round(mean_total_tokens, 1),
        # 상한 N 선정 근거: 배치 1회가 상한까지 LLM을 쓸 때의 예상 총 토큰
        "projected_tokens_per_batch": round(mean_total_tokens * _CAP, 1),
    }

    live_run_recorder["commit_analysis_cost"] = summary
    print("\n=== 커밋 분석 비용 실측(L5) ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

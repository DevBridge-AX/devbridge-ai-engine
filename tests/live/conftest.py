"""
라이브 E2E 검증 하네스(tests/live) 공통 fixture.

게이트: RUN_LIVE_LLM=1 그리고 settings.gms_api_key가 비어있지 않을 때만 이 디렉터리의
테스트를 실행합니다(autouse 세션 fixture `_require_live_env`). 키 존재만으로는 켜지지
않습니다 — .env가 자동 로드되므로 GMS_API_KEY가 이미 있어도 RUN_LIVE_LLM=1을 명시해야
과금이 발생합니다. `--collect-only`는 fixture를 실행하지 않으므로 테스트 수집 자체는
게이트와 무관하게 항상 가능합니다.

세션 1회(`live_env`): tmp VECTOR_STORE_PATH/METRICS_DIR를 지정하고
MAIN_MAX_TOKENS=256으로 비용을 제한한 뒤 get_settings()/get_vector_store() lru_cache를
초기화합니다. `seeded_workspace`가 tests/live/fixtures/corpus를 이 tmp 워크스페이스에
실 임베딩으로 인덱싱합니다(scripts/eval/seed.py::seed_workspace, embedder 라이브 검증 겸함).
`client`는 이 워크스페이스가 보이는 TestClient(app)를 제공합니다.

세션 종료(`pytest_sessionfinish`): 이번 실행에서 기록된 llm_calls.jsonl/chat_metrics.jsonl을
purpose×model별 토큰 합계/지연 분위수로 집계해 터미널에 출력하고
data/live_runs/{YYYYMMDD-HHMMSS}.json에 저장합니다. 라이브 테스트가 실제로 실행되어
`live_env`가 셋업된 세션에서만 동작합니다.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scripts.metrics_report import (
    compute_llm_calls_breakdown,
    compute_stage_percentiles,
    load_records,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# pytest_sessionfinish는 fixture가 아니라 세션 훅이라 fixture 값을 직접 받을 수 없으므로,
# live_env/seeded_workspace가 이 모듈 전역에 자신이 만든 값을 남겨 훅이 읽습니다.
# live_env가 셋업되지 않은 세션(=라이브 미실행)에서는 비어 있고, 이때 훅은 아무 것도 하지 않습니다.
_LIVE_STATE: dict = {}


def _live_enabled() -> bool:
    """RUN_LIVE_LLM=1 AND settings.gms_api_key 비어있지 않음을 확인합니다."""
    if os.environ.get("RUN_LIVE_LLM") != "1":
        return False

    from app.config import get_settings

    return bool(get_settings().gms_api_key)


@pytest.fixture(scope="session", autouse=True)
def _require_live_env():
    """RUN_LIVE_LLM=1 + gms_api_key 조건을 만족하지 않으면 tests/live 전체를 skip합니다."""
    if not _live_enabled():
        pytest.skip(
            "RUN_LIVE_LLM=1 and a non-empty GMS_API_KEY are required to run tests/live "
            "(docs/ai-api-usage.md '라이브 검증 실행법' 참고)"
        )


@pytest.fixture(scope="session")
def live_env(tmp_path_factory):
    """tmp VECTOR_STORE_PATH/METRICS_DIR 지정 + MAIN_MAX_TOKENS=256 + 설정/벡터스토어 캐시 초기화."""
    from app.config import get_settings
    from app.db.vector_store import get_vector_store

    base = tmp_path_factory.mktemp("live")
    vector_store_path = base / "vector_store"
    metrics_dir = base / "metrics"
    vector_store_path.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    os.environ["VECTOR_STORE_PATH"] = str(vector_store_path)
    os.environ["METRICS_DIR"] = str(metrics_dir)
    os.environ["MAIN_MAX_TOKENS"] = "256"

    get_settings.cache_clear()
    get_vector_store.cache_clear()

    _LIVE_STATE["metrics_dir"] = metrics_dir

    yield {"base": base, "vector_store_path": vector_store_path, "metrics_dir": metrics_dir}

    get_settings.cache_clear()
    get_vector_store.cache_clear()


@pytest.fixture(scope="session")
def seeded_workspace(live_env, tmp_path_factory):
    """tests/live/fixtures/corpus를 실 임베딩으로 인덱싱한 (db, workspace_id)를 반환합니다."""
    from scripts.eval.seed import seed_workspace

    tmp_dir = tmp_path_factory.mktemp("live_db")
    db, workspace_id = seed_workspace(tmp_dir)
    _LIVE_STATE["workspace_id"] = workspace_id

    yield db, workspace_id

    db.close()


@pytest.fixture(scope="session")
def client(seeded_workspace):
    """X-Internal-Api-Key 헤더가 설정된 TestClient(app). get_db는 시드된 db로 override."""
    from app.config import get_settings
    from app.db.session import get_db
    from app.main import app

    db, _ = seeded_workspace

    def _override_get_db():
        yield db

    app.dependency_overrides[get_db] = _override_get_db
    settings = get_settings()

    with TestClient(app) as test_client:
        test_client.headers.update({"X-Internal-Api-Key": settings.internal_api_key})
        yield test_client

    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(scope="session")
def live_run_recorder():
    """페르소나 답변 앞 200자 등 수동 검토용 데이터를 세션 종료 리포트에 실어 보냅니다."""
    return _LIVE_STATE.setdefault("persona_answers", {})


def pytest_sessionfinish(session, exitstatus):
    """라이브 실행 시에만 llm_calls/chat_metrics를 집계해 출력하고 run 파일로 저장합니다."""
    metrics_dir = _LIVE_STATE.get("metrics_dir")
    if metrics_dir is None:
        return  # live_env가 셋업되지 않음 = 이번 세션은 라이브 테스트를 실행하지 않음

    records_by_event = load_records(Path(metrics_dir))
    llm_calls = records_by_event.get("llm_calls", [])
    chat_metrics = records_by_event.get("chat_metrics", [])

    token_sums: dict[str, dict[str, int]] = {}
    for record in llm_calls:
        key = f"{record.get('purpose') or 'unknown'}×{record.get('model') or 'unknown'}"
        agg = token_sums.setdefault(
            key, {"prompt_tokens": 0, "completion_tokens": 0, "thoughts_tokens": 0}
        )
        agg["prompt_tokens"] += int(record.get("prompt_tokens") or 0)
        agg["completion_tokens"] += int(record.get("completion_tokens") or 0)
        agg["thoughts_tokens"] += int(record.get("thoughts_tokens") or 0)

    breakdown = compute_llm_calls_breakdown(llm_calls)
    latency_breakdown = {
        f"{purpose}×{model}": {
            "count": stats["count"],
            "p50_latency_ms": round(stats["p50_latency_ms"], 1),
            "p95_latency_ms": round(stats["p95_latency_ms"], 1),
        }
        for (purpose, model), stats in breakdown.items()
    }

    chat_stage_percentiles = compute_stage_percentiles(chat_metrics)
    chat_stage_latency_ms = {
        stage: {"p50": round(p50, 1), "p95": round(p95, 1)}
        for stage, (p50, p95) in chat_stage_percentiles.items()
    }

    report = {
        "timestamp": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "workspace_id": _LIVE_STATE.get("workspace_id"),
        "llm_calls_count": len(llm_calls),
        "chat_requests_count": len(chat_metrics),
        "token_sums_by_purpose_model": token_sums,
        "llm_calls_latency": latency_breakdown,
        "chat_stage_latency_ms": chat_stage_latency_ms,
        "persona_answers_preview": _LIVE_STATE.get("persona_answers", {}),
    }

    report_text = json.dumps(report, ensure_ascii=False, indent=2)
    print("\n=== 라이브 검증 토큰/지연 집계 ===")
    print(report_text)

    out_dir = _REPO_ROOT / "data" / "live_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{report['timestamp']}.json"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\n결과 파일: {out_path}")

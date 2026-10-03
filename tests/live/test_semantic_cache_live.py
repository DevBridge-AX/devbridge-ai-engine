"""
시맨틱 캐시 hit 경로 라이브 측정 (실 GMS API 호출, 과금 발생).

RUN_LIVE_LLM=1 uv run --extra dev python -m pytest -m live -q tests/live/test_semantic_cache_live.py
로 실행합니다(docs/semantic-cache-eval.md 참고). 바꿔 말한 질문의 적중률은 오프라인 스윕
(scripts/eval/cache_eval.py)이 더 신뢰할 수 있으므로, 이 테스트는 다음 두 가지만 측정합니다.
(a) 동일 질문 반복 시 캐시 hit 경로의 지연/토큰 절감 효과
(b) 기본 임계치(0.95)에서 오프라인 스윕상 유일하게 적중하는 paraphrase 1건(p001)의 적중 여부

LLM 호출은 Phase 1의 8개 질문(질문당 grounding + main)에 한정됩니다. 단언은 응답 계약과
캐시 경로의 불변식(hit 시 토큰 0, grounding_stage="cache" 등)만 다루며, paraphrase 적중
여부는 실패 조건이 아니라 결과로 기록합니다. 결과는 `live_run_recorder["semantic_cache"]`와
터미널 출력으로 남깁니다.

재검증(LLM verify) on 변형: `test_semantic_cache_verify_path`는 후보 구간(0.80~0.95)의 paraphrase가
재검증을 거쳐 hit 되는 경로(목표: hit 경로 total_ms p50 <= 2s)와 NO -> miss 경로 비용을 측정합니다.
이 변형만 실행하려면

RUN_LIVE_LLM=1 uv run --extra dev python -m pytest -m live -q tests/live/test_semantic_cache_live.py -k verify

호출 예산: main/grounding 각 8회(q1 8건 miss) + q2 miss 건수만큼 추가(최대 8회),
재검증(REWRITE_MODEL) 최대 16회(q2 8건 + q1끼리 후보 구간에 든 경우, 요청당 최대 1회). 결과는
`live_run_recorder["semantic_cache_verify"]`와 터미널 출력으로 남깁니다.
"""

import json
import os
from pathlib import Path

import pytest

from app.config import get_settings
from app.core.cache.semantic_cache import get_semantic_cache
from app.schemas.chat import ChatDoneEvent
from scripts.eval.cache_eval_metrics import load_dataset
from scripts.metrics_report import _percentile
from tests.live.sse import parse_sse

pytestmark = pytest.mark.live

_DATASET = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "datasets" / "cache_pairs.jsonl"
_PAIR_IDS = ["p001", "p006", "p008", "p010", "p011", "p016", "p017", "p019"]
_PARAPHRASE_ID = "p001"

# 재검증 on 변형 데이터. 2회차 오프라인 측정(docs/semantic-cache-eval.md)에서 이 same_intent 쌍들이
# 후보 구간 0.80~0.95에 들었고 재검증 결과가 YES였습니다.
_VERIFY_PAIR_IDS = ["p004", "p007", "p008", "p015", "p016", "p017", "p018"]
# 재검증 NO 사례(p011: 유사도 0.894, v1 프롬프트에서 verify NO) - NO -> miss 경로 비용 측정용
_VERIFY_NO_PAIR_IDS = ["p011"]
_VERIFY_THRESHOLD = 0.95
_VERIFY_CANDIDATE_THRESHOLD = 0.80


@pytest.fixture(scope="module")
def semantic_cache_on(live_env):
    """SEMANTIC_CACHE_ENABLED=true로 설정/캐시 싱글톤을 초기화하고 종료 시 복원합니다."""
    previous = os.environ.get("SEMANTIC_CACHE_ENABLED")
    os.environ["SEMANTIC_CACHE_ENABLED"] = "true"
    get_settings.cache_clear()
    get_semantic_cache.cache_clear()
    get_semantic_cache().clear()

    yield get_semantic_cache()

    get_semantic_cache().clear()
    if previous is None:
        os.environ.pop("SEMANTIC_CACHE_ENABLED", None)
    else:
        os.environ["SEMANTIC_CACHE_ENABLED"] = previous
    get_settings.cache_clear()
    get_semantic_cache.cache_clear()


@pytest.fixture
def semantic_cache_verify_on(live_env):
    """재검증 on 설정(함수 스코프). 시작/종료 시 캐시를 비워 다른 테스트의 항목과 섞이지 않게 합니다."""
    overrides = {
        "SEMANTIC_CACHE_ENABLED": "true",
        "SEMANTIC_CACHE_VERIFY_ENABLED": "true",
        "SEMANTIC_CACHE_CANDIDATE_THRESHOLD": str(_VERIFY_CANDIDATE_THRESHOLD),
    }
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    get_settings.cache_clear()
    get_semantic_cache.cache_clear()
    get_semantic_cache().clear()

    yield get_semantic_cache()

    get_semantic_cache().clear()
    for key, value in previous.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    get_settings.cache_clear()
    get_semantic_cache.cache_clear()


def _last_chat_metric() -> dict:
    path = Path(os.environ["METRICS_DIR"]) / "chat_metrics.jsonl"
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1])


def _llm_call_records() -> list[dict]:
    path = Path(os.environ["METRICS_DIR"]) / "llm_calls.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


_counter = 0


def _ask(client, workspace_id: str, question: str) -> dict:
    """질문 1건을 요청하고 answer/done/metric/ChatDoneEvent 검증 결과를 반환합니다."""
    global _counter
    _counter += 1
    calls_before = len(_llm_call_records())
    resp = client.post(
        "/api/chat/",
        json={
            "session_id": f"live-cache-{_counter}",
            "content": question,
            "conversation_history": [],
            "workspace_id": workspace_id,
            "user_id": "live-user",
            "role": "developer",
        },
    )
    assert resp.status_code == 200
    events = parse_sse(resp.text)
    assert events and events[-1][0] == "done"
    assert sum(1 for name, _ in events if name == "error") == 0
    answer = "".join(data.get("text", "") for name, data in events if name == "token")
    done = ChatDoneEvent.model_validate(events[-1][1])
    # 재검증 호출 토큰은 token_usage에 넣지 않고 llm_calls(purpose="cache_verify")에만 기록됩니다.
    verify_calls = [
        c for c in _llm_call_records()[calls_before:] if c.get("purpose") == "cache_verify"
    ]
    return {
        "answer": answer,
        "done": done,
        "metric": _last_chat_metric(),
        "verify_calls": verify_calls,
    }


def _llm_tokens(metric: dict) -> int:
    return sum(
        int(metric.get(key) or 0)
        for key in (
            "main_prompt_tokens",
            "main_completion_tokens",
            "grounding_prompt_tokens",
            "grounding_completion_tokens",
        )
    )


def _p(values: list[float]) -> dict:
    if not values:
        return {"p50": None, "p95": None}
    return {"p50": round(_percentile(values, 50), 1), "p95": round(_percentile(values, 95), 1)}


def test_semantic_cache_hit_path(client, seeded_workspace, semantic_cache_on, live_run_recorder):
    _, workspace_id = seeded_workspace
    pairs = {p["id"]: p for p in load_dataset(_DATASET)}
    questions = {pid: pairs[pid]["q1"] for pid in _PAIR_IDS}

    # Phase 1: miss (기준선)
    phase1: dict[str, dict] = {}
    for pid, question in questions.items():
        result = _ask(client, workspace_id, question)
        assert result["metric"]["cache_hit"] is False
        phase1[pid] = result
    # 그라운딩 실패/빈 답변은 캐시에 저장되지 않으므로 비교 대상에서 제외하고 기록만 남깁니다.
    comparable = [
        pid for pid, r in phase1.items() if r["done"].is_groundable and r["answer"].strip()
    ]
    excluded = [pid for pid in phase1 if pid not in comparable]

    # Phase 2: hit (동일 질문 재요청)
    phase2: dict[str, dict] = {}
    for pid in comparable:
        result = _ask(client, workspace_id, questions[pid])
        base = phase1[pid]
        assert result["metric"]["cache_hit"] is True
        assert result["metric"]["grounding_stage"] == "cache"
        usage = result["done"].token_usage
        assert usage.main.prompt_tokens == 0 and usage.main.completion_tokens == 0
        assert usage.grounding is None
        assert result["answer"] == base["answer"]
        assert result["done"].citations == base["done"].citations
        phase2[pid] = result

    # Phase 3: paraphrase (오프라인 스윕에서 0.95 이상인 유일한 same_intent 쌍)
    paraphrase = _ask(client, workspace_id, pairs[_PARAPHRASE_ID]["q2"])
    paraphrase_metric = paraphrase["metric"]
    paraphrase_hit = bool(paraphrase_metric["cache_hit"])
    paraphrase_similarity = paraphrase_metric.get("cache_similarity")

    # 집계
    def collect(results: dict[str, dict], key: str) -> list[float]:
        return [float(r["metric"][key]) for r in results.values() if r["metric"].get(key) is not None]

    base_results = {pid: phase1[pid] for pid in comparable}
    p1_total, p2_total = collect(base_results, "total_ms"), collect(phase2, "total_ms")
    p1_ttft, p2_ttft = collect(base_results, "ttft_ms"), collect(phase2, "ttft_ms")
    tokens_p1 = [_llm_tokens(r["metric"]) for r in base_results.values()]
    mean_tokens_p1 = sum(tokens_p1) / len(tokens_p1) if tokens_p1 else None
    mean_tokens_p2 = (
        sum(_llm_tokens(r["metric"]) for r in phase2.values()) / len(phase2) if phase2 else None
    )
    saving = (
        (mean_tokens_p1 - mean_tokens_p2) / mean_tokens_p1 if mean_tokens_p1 else None
    )
    p1_total_p50 = _percentile(p1_total, 50) if p1_total else None
    p2_total_p50 = _percentile(p2_total, 50) if p2_total else None
    speedup = (
        round(p1_total_p50 / p2_total_p50, 1) if p1_total_p50 and p2_total_p50 else None
    )

    summary = {
        "threshold": get_settings().semantic_cache_threshold,
        "questions": len(questions),
        "comparable": len(comparable),
        "excluded_not_groundable": excluded,
        "phase1_miss": {"total_ms": _p(p1_total), "ttft_ms": _p(p1_ttft)},
        "phase2_hit": {"total_ms": _p(p2_total), "ttft_ms": _p(p2_ttft)},
        "mean_llm_tokens_per_request": {"phase1": mean_tokens_p1, "phase2": mean_tokens_p2},
        "token_saving_rate": None if saving is None else round(saving, 4),
        "hit_latency_speedup_total_p50": speedup,
        "paraphrase": {
            "id": _PARAPHRASE_ID,
            "paraphrase_hit": paraphrase_hit,
            "similarity": paraphrase_similarity,
        },
    }
    live_run_recorder["semantic_cache"] = summary
    print("\n=== 시맨틱 캐시 라이브 측정 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 1) if values else None


def test_semantic_cache_verify_path(
    client, seeded_workspace, semantic_cache_verify_on, live_run_recorder
):
    """재검증 on 상태에서 후보 구간 paraphrase의 hit/miss 경로 지연과 토큰을 측정합니다."""
    _, workspace_id = seeded_workspace
    settings = get_settings()
    pairs = {p["id"]: p for p in load_dataset(_DATASET)}
    pair_ids = _VERIFY_PAIR_IDS + _VERIFY_NO_PAIR_IDS

    # Phase 1: q1 miss (캐시 시드, 기준선)
    # 후보 하한이 0.80이라 주제가 겹치는 q1끼리(예: 결제 API 질문들) 후보 구간에 들어 재검증이
    # 호출될 수 있습니다. 이때 hit 된 쌍은 시드가 되지 않으므로 실패가 아니라 기록 후 제외합니다.
    phase1: dict[str, dict] = {}
    for pid in pair_ids:
        phase1[pid] = _ask(client, workspace_id, pairs[pid]["q1"])
    phase1_hits = [pid for pid, r in phase1.items() if r["metric"]["cache_hit"]]
    phase1_verify_calls = sum(len(r["verify_calls"]) for r in phase1.values())
    # 그라운딩 실패/빈 답변은 캐시에 저장되지 않으므로 비교 대상에서 제외하고 기록만 남깁니다.
    comparable = [
        pid
        for pid, r in phase1.items()
        if pid not in phase1_hits and r["done"].is_groundable and r["answer"].strip()
    ]
    excluded = [pid for pid in phase1 if pid not in comparable and pid not in phase1_hits]

    # Phase 2: q2 paraphrase (재검증 경로). 결과(hit/miss)는 실패 조건이 아니라 기록 대상입니다.
    phase2: dict[str, dict] = {}
    rows: list[dict] = []
    for pid in comparable:
        result = _ask(client, workspace_id, pairs[pid]["q2"])
        metric = result["metric"]
        hit = bool(metric["cache_hit"])
        similarity = metric.get("cache_similarity")
        verify_result = metric.get("cache_verify_result")
        verify_ms = metric.get("cache_verify_ms")
        usage = result["done"].token_usage

        if hit and similarity is not None and similarity < _VERIFY_THRESHOLD:
            assert verify_result == "yes"
            assert verify_ms is not None
            assert usage.main.prompt_tokens == 0 and usage.main.completion_tokens == 0
            assert metric["grounding_stage"] == "cache"
        if hit and similarity is not None and similarity >= _VERIFY_THRESHOLD:
            assert verify_result is None
        if verify_result in ("no", "invalid", "error", "timeout"):
            assert hit is False

        phase2[pid] = result
        rows.append(
            {
                "id": pid,
                "similarity": similarity,
                "verify_result": verify_result,
                "verify_ms": verify_ms,
                "hit": hit,
                "total_ms": metric.get("total_ms"),
            }
        )

    def vals(results: list[dict], key: str) -> list[float]:
        return [float(r["metric"][key]) for r in results if r["metric"].get(key) is not None]

    hit_results = [r for r in phase2.values() if r["metric"]["cache_hit"]]
    verified_hits = [r for r in hit_results if r["metric"].get("cache_verify_result") == "yes"]
    miss_after_no = [
        r for r in phase2.values() if r["metric"].get("cache_verify_result") == "no"
    ]
    hit_total = vals(hit_results, "total_ms")
    hit_total_p50 = _percentile(hit_total, 50) if hit_total else None
    verify_calls = [c for r in phase2.values() for c in r["verify_calls"]]
    verify_tokens = [
        int(c.get("prompt_tokens") or 0) + int(c.get("completion_tokens") or 0)
        for c in verify_calls
    ]

    summary = {
        "settings": {
            "threshold": settings.semantic_cache_threshold,
            "candidate_threshold": settings.semantic_cache_candidate_threshold,
            "verify_prompt_version": settings.semantic_cache_verify_prompt_version,
            "rewrite_model": settings.rewrite_model,
        },
        "pairs": len(pair_ids),
        "comparable": len(comparable),
        "excluded_not_groundable": excluded,
        "phase1_cache_hits": phase1_hits,
        "phase1_verify_calls": phase1_verify_calls,
        "rows": rows,
        "hit_count": len(hit_results),
        "verified_hit_count": len(verified_hits),
        "hit_path": {
            "total_ms": _p(hit_total),
            "ttft_ms": _p(vals(hit_results, "ttft_ms")),
            "verify_ms": _p(vals(hit_results, "cache_verify_ms")),
        },
        "verify_ms_all": _p(vals(list(phase2.values()), "cache_verify_ms")),
        "miss_after_no_total_ms": [r["metric"].get("total_ms") for r in miss_after_no],
        "mean_llm_tokens_per_request": {
            "phase1": _mean([_llm_tokens(phase1[pid]["metric"]) for pid in comparable]),
            "phase2": _mean([_llm_tokens(r["metric"]) for r in phase2.values()]),
        },
        "verify_calls": len(verify_calls),
        "mean_verify_tokens_per_call": _mean(verify_tokens),
        "meets_l6": hit_total_p50 is not None and hit_total_p50 <= 2000,
    }
    live_run_recorder["semantic_cache_verify"] = summary
    print("\n=== 시맨틱 캐시 재검증 on 라이브 측정 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

"""
그라운딩 임계치 오프라인 평가 (A4).

목적: app/core/rag/grounding.py의 1차 유사도 필터(grounding_similarity_threshold)와
LLM 2차 판정을 라벨셋(scripts/eval/datasets/grounding_cases.jsonl)으로 측정해
임계치 조정과 fail-open 계약의 근거 데이터를 만듭니다. 이 스크립트 자체는 config의
기본 임계치를 변경하지 않습니다 — 값 변경은 이 결과를 근거로 한 별도 결정입니다.

이 모듈은 두 부분으로 나뉩니다.
- 순수 함수(compute_metrics/render_sweep_markdown/load_dataset/load_cache/save_cache):
  API/DB 호출이 없어 tests/test_grounding_eval_metrics.py가 합성 레코드로 단위 테스트합니다.
- 라이브 수집(CLI --live): 실 GMS API를 호출합니다(비용 발생). 아래 CLI 사용법을 참고하세요.

사용법:
    # 1) 라이브 수집 (실 GMS API 호출, 비용 발생) — tmp 워크스페이스를 시드하고
    #    케이스마다 retriever.retrieve() + (임계치 무관) call_grounding()을 호출해
    #    결과를 캐시 jsonl로 저장합니다.
    RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live
    RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live --limit 5   # 스모크용 소량 실행

    # 1-b) 판정 변형 비교(G1) — 케이스당 검색 1회 + 변형마다 판정 1회(변형 수만큼 비용 증가).
    #      --variants 기본값은 VARIANTS 전체(baseline,strict,top3,strict_top3,strict_top3_cap).
    RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live --variants baseline,strict

    # 1-c) 재현성 측정(L4) — 변형마다 판정을 N번 반복. 검색은 케이스당 1회(결정적),
    #      LLM 호출 수 = 케이스 수 x 변형 수 x N (비용 N배). --live에서만 사용 가능.
    #      예: 40케이스 x 2변형 x 2회 = 160회.
    RUN_LIVE_LLM=1 python3 scripts/eval/grounding_eval.py --live --variants baseline,strict_top3 --repeat 2

    # 2) 오프라인 스윕 (API 호출 없음) — 캐시된 결과로 임계치 0.20~0.60을 스윕합니다.
    #    캐시에 변형이 여러 개면 baseline 스윕에 더해 변형 비교 표(임계치 기본값은
    #    settings.grounding_similarity_threshold, --threshold로 override)와 변형 간 판정이
    #    갈린 케이스 표를 출력합니다. variant 필드가 없는 레코드(A4 캐시)는 baseline입니다.
    #    캐시에 repeat가 2개 이상이면 위 섹션은 repeat 0 레코드만 쓰고, 끝에 반복 측정
    #    섹션(variant별 mean±sd/min~max, 판정 불안정 케이스)을 덧붙입니다.
    python3 scripts/eval/grounding_eval.py --from-cache data/eval/grounding-20260930-120000.jsonl
    python3 scripts/eval/grounding_eval.py --from-cache data/eval/grounding-....jsonl --threshold 0.35

캐시 레코드 스키마(jsonl 1줄 = 케이스x변형 1건):
    {variant, id, category, expected_groundable, top_similarity, llm_is_groundable,
     llm_confidence, parse_ok, fallback_reason, latency_ms, prompt_tokens,
     completion_tokens, prompt_chars, repeat}
    (variant/prompt_chars는 G1, repeat(0-based 반복 인덱스)는 L4에서 추가된 필드이며
     없어도 로드됩니다. repeat가 없는 레코드는 repeat 0으로 취급합니다.)

fail-open 반영: app/core/rag/grounding.py::assess()는 call_grounding 예외
("llm_error") 또는 파싱 실패("parse_error") 시 is_groundable=True로 fail-open
합니다. 이 스크립트는 임계치와 무관하게 항상 call_grounding을 호출하므로, 위 두
경우에도 llm_is_groundable/llm_confidence를 기록하지 못합니다(parse_ok=False,
fallback_reason만 남음). compute_metrics()는 이 fallback 레코드를 실제 앱과 동일하게
"LLM 단계까지 도달했다면 항상 groundable=True"로 취급합니다.
"""

import argparse
import asyncio
import json
import statistics
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "scripts" / "eval" / "datasets" / "grounding_cases.jsonl"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "eval"

# 오프라인 스윕 임계치: 0.20 ~ 0.60, 0.05 간격 (부동소수 오차 방지를 위해 정수 스텝으로 계산)
SWEEP_THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(9)]

# 판정 변형(G1): 프롬프트 버전 / 판정에 넣는 청크 수 / 청크 길이 상한(0=제한 없음).
# baseline은 W1 이전 기본값(v1 / 5)을 명시적으로 고정한 것이며(현재 기본값은 strict_top3와 동일),
# 이 표는 app/config.py 기본값과 무관한 비교 실험 전용입니다.
VARIANTS: dict[str, dict] = {
    "baseline": {"prompt_version": "v1", "top_k": 5, "max_chunk_chars": 0},
    "strict": {"prompt_version": "v2-strict", "top_k": 5, "max_chunk_chars": 0},
    "top3": {"prompt_version": "v1", "top_k": 3, "max_chunk_chars": 0},
    "strict_top3": {"prompt_version": "v2-strict", "top_k": 3, "max_chunk_chars": 0},
    "strict_top3_cap": {"prompt_version": "v2-strict", "top_k": 3, "max_chunk_chars": 600},
}

_CONFIDENCE_BUCKETS = [("0.0-0.5", 0.0, 0.5), ("0.5-0.8", 0.5, 0.8), ("0.8-1.0", 0.8, 1.0)]


# ---------------------------------------------------------------------------
# 데이터셋 / 캐시 I/O (순수 함수 — API/DB 호출 없음)
# ---------------------------------------------------------------------------

def load_dataset(path: Path) -> list[dict]:
    """scripts/eval/datasets/grounding_cases.jsonl 형식의 평가 케이스를 읽습니다."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    return cases


def load_cache(path: Path) -> list[dict]:
    """--live로 생성된 결과 캐시(jsonl)를 읽습니다. 파싱 실패한 줄은 건너뜁니다."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def save_cache(path: Path, records: list[dict]) -> None:
    """수집된 레코드를 jsonl로 저장합니다(1줄 = 케이스 1건)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# 집계 (순수 함수 — API/DB 호출 없음, tests/test_grounding_eval_metrics.py 대상)
# ---------------------------------------------------------------------------

def _effective_llm_groundable(record: dict) -> bool:
    """fail-open 계약(app/core/rag/grounding.py::assess())을 반영한 LLM 2차 판정 값.

    fallback_reason(llm_error/parse_error)이 있으면 실제 llm_is_groundable 값과
    무관하게 항상 True(fail-open)입니다. fallback이 없으면 기록된 값을 그대로 씁니다.
    """
    if record.get("fallback_reason") is not None:
        return True
    return bool(record.get("llm_is_groundable"))


def _final_decision(record: dict, threshold: float) -> bool:
    """assess()와 동일한 결합 로직: 1차 필터(top_similarity >= threshold) 통과 시에만
    LLM 2차 판정(fail-open 반영)으로 최종 groundable 여부를 정합니다."""
    if float(record.get("top_similarity", 0.0)) < threshold:
        return False
    return _effective_llm_groundable(record)


def compute_metrics(records: list[dict], threshold: float) -> dict:
    """레코드 목록에 threshold를 적용했을 때의 그라운딩 판정 품질 지표를 계산합니다.

    순수 함수입니다 — --from-cache가 임계치를 스윕할 때마다 API 재호출 없이 이 함수만
    반복 호출합니다.

    Returns:
        {
          "threshold": float, "total": int,
          "accuracy": float | None,
          "not_groundable": {"precision", "recall", "f1"},   # not-groundable = positive class
          "filter_false_block_rate": {"overall", "by_category"},  # 라벨 groundable인데 sim<threshold
          "llm_call_saving_rate": float | None,   # top_similarity < threshold 비율
          "llm_only_agreement": float | None,     # 1차 필터 무시, LLM 2차 판정(fail-open 반영) 단독 정답률
          "parse_failure_rate": float | None,
          "confidence_bucket_accuracy": {"0.0-0.5", "0.5-0.8", "0.8-1.0"},
          "mean_prompt_tokens", "mean_completion_tokens", "mean_latency_ms",
        }
    """
    total = len(records)
    if total == 0:
        return {
            "threshold": threshold,
            "total": 0,
            "accuracy": None,
            "not_groundable": {"precision": None, "recall": None, "f1": None},
            "filter_false_block_rate": {"overall": None, "by_category": {}},
            "llm_call_saving_rate": None,
            "llm_only_agreement": None,
            "parse_failure_rate": None,
            "confidence_bucket_accuracy": {name: None for name, _, _ in _CONFIDENCE_BUCKETS},
            "mean_prompt_tokens": None,
            "mean_completion_tokens": None,
            "mean_latency_ms": None,
        }

    correct = 0
    tp = fp = fn = 0  # not-groundable(final=False)을 positive class로

    total_groundable = 0
    false_block_count = 0
    false_block_by_category: dict[str, list[int]] = {}  # cat -> [false_block_n, groundable_n]

    below_threshold_count = 0
    llm_only_agree = 0
    parse_failures = 0

    bucket_correct = {name: 0 for name, _, _ in _CONFIDENCE_BUCKETS}
    bucket_total = {name: 0 for name, _, _ in _CONFIDENCE_BUCKETS}

    latencies: list[float] = []
    prompt_tokens_list: list[float] = []
    completion_tokens_list: list[float] = []

    for record in records:
        expected = bool(record["expected_groundable"])
        top_similarity = float(record.get("top_similarity", 0.0))
        category = str(record.get("category", "unknown"))

        final = _final_decision(record, threshold)
        if final == expected:
            correct += 1

        predicted_not_groundable = not final
        actual_not_groundable = not expected
        if predicted_not_groundable and actual_not_groundable:
            tp += 1
        elif predicted_not_groundable and not actual_not_groundable:
            fp += 1
        elif not predicted_not_groundable and actual_not_groundable:
            fn += 1

        if expected:
            total_groundable += 1
            cat_counts = false_block_by_category.setdefault(category, [0, 0])
            cat_counts[1] += 1
            if top_similarity < threshold:
                false_block_count += 1
                cat_counts[0] += 1

        if top_similarity < threshold:
            below_threshold_count += 1

        effective = _effective_llm_groundable(record)
        if effective == expected:
            llm_only_agree += 1

        if not record.get("parse_ok", True):
            parse_failures += 1

        confidence = record.get("llm_confidence")
        if record.get("parse_ok") and confidence is not None:
            for name, lo, hi in _CONFIDENCE_BUCKETS:
                if lo <= confidence < hi or (hi == 1.0 and confidence == hi):
                    bucket_total[name] += 1
                    if effective == expected:
                        bucket_correct[name] += 1
                    break

        if record.get("latency_ms") is not None:
            latencies.append(float(record["latency_ms"]))
        if record.get("prompt_tokens") is not None:
            prompt_tokens_list.append(float(record["prompt_tokens"]))
        if record.get("completion_tokens") is not None:
            completion_tokens_list.append(float(record["completion_tokens"]))

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )

    by_category_rate = {
        cat: (counts[0] / counts[1] if counts[1] else None)
        for cat, counts in sorted(false_block_by_category.items())
    }

    confidence_bucket_accuracy = {
        name: (bucket_correct[name] / bucket_total[name] if bucket_total[name] else None)
        for name, _, _ in _CONFIDENCE_BUCKETS
    }

    def _mean(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "threshold": threshold,
        "total": total,
        "accuracy": correct / total,
        "not_groundable": {"precision": precision, "recall": recall, "f1": f1},
        "filter_false_block_rate": {
            "overall": (false_block_count / total_groundable) if total_groundable else None,
            "by_category": by_category_rate,
        },
        "llm_call_saving_rate": below_threshold_count / total,
        "llm_only_agreement": llm_only_agree / total,
        "parse_failure_rate": parse_failures / total,
        "confidence_bucket_accuracy": confidence_bucket_accuracy,
        "mean_prompt_tokens": _mean(prompt_tokens_list),
        "mean_completion_tokens": _mean(completion_tokens_list),
        "mean_latency_ms": _mean(latencies),
    }


def group_by_variant(records: list[dict]) -> dict[str, list[dict]]:
    """레코드를 variant별로 묶습니다. variant 필드가 없는 레코드(A4 캐시)는 "baseline"입니다."""
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("variant") or "baseline"), []).append(record)
    return grouped


def compute_variant_comparison(
    records_by_variant: dict[str, list[dict]], threshold: float
) -> dict:
    """variant별 지표와 variant 간 판정이 갈린 케이스를 계산합니다(순수 함수, API 호출 없음).

    Returns:
        {
          "threshold": float,
          "variants": {name: {"total", "accuracy", "precision", "recall", "f1",
                              "groundable_false_block_count", "groundable_total",
                              "groundable_false_block_rate", "mean_prompt_tokens",
                              "mean_latency_ms", "parse_failure_rate"}},
          "disagreements": [{"id", "category", "expected", "correct": {variant: bool|None}}],
        }
    groundable 오차단 = expected_groundable=True인데 최종 판정이 False인 건
    (1차 필터 차단 + 2차 판정 false 모두 포함).
    """
    variants: dict[str, dict] = {}
    finals: dict[str, dict[str, tuple[bool, dict]]] = {}  # variant -> case id -> (final, record)

    for name, records in records_by_variant.items():
        m = compute_metrics(records, threshold)
        ng = m["not_groundable"]
        groundable_total = 0
        false_block = 0
        per_case: dict[str, tuple[bool, dict]] = {}
        for record in records:
            final = _final_decision(record, threshold)
            per_case[str(record["id"])] = (final, record)
            if bool(record["expected_groundable"]):
                groundable_total += 1
                if not final:
                    false_block += 1
        finals[name] = per_case
        variants[name] = {
            "total": m["total"],
            "accuracy": m["accuracy"],
            "precision": ng["precision"],
            "recall": ng["recall"],
            "f1": ng["f1"],
            "groundable_false_block_count": false_block,
            "groundable_total": groundable_total,
            "groundable_false_block_rate": (false_block / groundable_total) if groundable_total else None,
            "mean_prompt_tokens": m["mean_prompt_tokens"],
            "mean_latency_ms": m["mean_latency_ms"],
            "parse_failure_rate": m["parse_failure_rate"],
        }

    case_ids: list[str] = []
    for per_case in finals.values():
        for cid in per_case:
            if cid not in case_ids:
                case_ids.append(cid)

    disagreements = []
    for cid in case_ids:
        present = {name: per_case[cid] for name, per_case in finals.items() if cid in per_case}
        if len({final for final, _ in present.values()}) <= 1:
            continue
        any_record = next(iter(present.values()))[1]
        expected = bool(any_record["expected_groundable"])
        disagreements.append(
            {
                "id": cid,
                "category": str(any_record.get("category", "unknown")),
                "expected": expected,
                "correct": {
                    name: (present[name][0] == expected if name in present else None)
                    for name in finals
                },
            }
        )

    return {"threshold": threshold, "variants": variants, "disagreements": disagreements}


def render_variant_markdown(records_by_variant: dict[str, list[dict]], threshold: float) -> str:
    """variant 비교 표와 판정이 갈린 케이스 표를 markdown으로 렌더링합니다(순수 함수)."""
    result = compute_variant_comparison(records_by_variant, threshold)
    lines = [
        f"# 그라운딩 판정 변형 비교 (threshold {threshold:.2f})",
        "",
        "| variant | accuracy | not-gr precision | not-gr recall | not-gr F1 "
        "| groundable 오차단 | 평균 prompt_tokens | 평균 latency_ms | 파싱 실패율 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, v in result["variants"].items():
        fb = (
            f"{v['groundable_false_block_count']}/{v['groundable_total']} "
            f"({_fmt(v['groundable_false_block_rate'])})"
        )
        lines.append(
            f"| {name} | {_fmt(v['accuracy'])} | {_fmt(v['precision'])} | {_fmt(v['recall'])} | "
            f"{_fmt(v['f1'])} | {fb} | {_fmt(v['mean_prompt_tokens'], pct=False)} | "
            f"{_fmt(v['mean_latency_ms'], pct=False)} | {_fmt(v['parse_failure_rate'])} |"
        )

    names = list(result["variants"])
    lines += ["", "## variant 간 판정이 갈린 케이스 (✓ 정답 / ✗ 오답)", ""]
    if not result["disagreements"]:
        lines.append("없음")
    else:
        lines.append("| id | category | expected | " + " | ".join(names) + " |")
        lines.append("| --- | --- | --- | " + " | ".join("---" for _ in names) + " |")
        for d in result["disagreements"]:
            expected = "groundable" if d["expected"] else "not-groundable"
            cells = " | ".join(
                "-" if d["correct"][n] is None else ("✓" if d["correct"][n] else "✗") for n in names
            )
            lines.append(f"| {d['id']} | {d['category']} | {expected} | {cells} |")

    return "\n".join(lines).rstrip() + "\n"


def group_by_repeat(records: list[dict]) -> dict[int, list[dict]]:
    """레코드를 repeat 인덱스별로 묶습니다. repeat 필드가 없는 레코드(구 캐시)는 0입니다."""
    grouped: dict[int, list[dict]] = {}
    for record in records:
        grouped.setdefault(int(record.get("repeat") or 0), []).append(record)
    return dict(sorted(grouped.items()))


_REPEAT_METRICS = (
    "accuracy",
    "recall",
    "f1",
    "groundable_false_block_rate",
    "mean_prompt_tokens",
    "mean_latency_ms",
    "parse_failure_rate",
)


def _summarize(values: list[float | None]) -> dict:
    """None을 제외한 값들의 mean/표본 stdev(n<2면 None)/min/max."""
    vals = [v for v in values if v is not None]
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals) if vals else None,
        "stdev": statistics.stdev(vals) if len(vals) >= 2 else None,
        "min": min(vals) if vals else None,
        "max": max(vals) if vals else None,
    }


def compute_repeat_stats(records: list[dict], threshold: float) -> dict:
    """variant별로 repeat마다 지표를 계산해 repeat 간 mean/stdev/min/max와 판정 안정성을 구합니다.

    순수 함수(API 호출 없음). compute_variant_comparison과 같은 지표 정의를 쓴다.

    Returns:
        {"threshold": float, "variants": {name: {
            "repeats": int,                  # 해당 variant의 distinct repeat 수
            "n": int,                        # 케이스 수(distinct id)
            "metrics": {accuracy|recall|f1|groundable_false_block_rate|mean_prompt_tokens|
                        mean_latency_ms|parse_failure_rate: {n, mean, stdev, min, max}},
            "unstable_case_ids": [str], "flip_rate": float | None,
        }}}
    stdev는 표본 표준편차(repeat 1개면 None). 불안정 케이스 = 해당 케이스의 repeat들 사이에서
    최종 판정(_final_decision)이 하나라도 다른 케이스이며, 그 케이스에 존재하는 repeat만 비교한다.
    """
    variants: dict[str, dict] = {}
    for name, v_records in group_by_variant(records).items():
        per_repeat: dict[str, list[float | None]] = {k: [] for k in _REPEAT_METRICS}
        for rep_records in group_by_repeat(v_records).values():
            m = compute_metrics(rep_records, threshold)
            groundable = [r for r in rep_records if bool(r["expected_groundable"])]
            fb = sum(1 for r in groundable if not _final_decision(r, threshold))
            per_repeat["accuracy"].append(m["accuracy"])
            per_repeat["recall"].append(m["not_groundable"]["recall"])
            per_repeat["f1"].append(m["not_groundable"]["f1"])
            per_repeat["groundable_false_block_rate"].append(fb / len(groundable) if groundable else None)
            per_repeat["mean_prompt_tokens"].append(m["mean_prompt_tokens"])
            per_repeat["mean_latency_ms"].append(m["mean_latency_ms"])
            per_repeat["parse_failure_rate"].append(m["parse_failure_rate"])

        decisions: dict[str, set[bool]] = {}
        for r in v_records:
            decisions.setdefault(str(r["id"]), set()).add(_final_decision(r, threshold))
        unstable = [cid for cid, d in decisions.items() if len(d) > 1]
        variants[name] = {
            "repeats": len(group_by_repeat(v_records)),
            "n": len(decisions),
            "metrics": {k: _summarize(v) for k, v in per_repeat.items()},
            "unstable_case_ids": unstable,
            "flip_rate": (len(unstable) / len(decisions)) if decisions else None,
        }
    return {"threshold": threshold, "variants": variants}


def render_repeat_markdown(records: list[dict], threshold: float) -> str:
    """반복 측정 결과(repeat 간 mean±sd, 판정 불안정 케이스)를 markdown으로 렌더링합니다(순수 함수)."""
    stats = compute_repeat_stats(records, threshold)
    n_repeats = len(group_by_repeat(records))
    lines = [
        f"# 반복 측정 ({n_repeats}회, threshold {threshold:.2f})",
        "",
        "mean±sd는 repeat 간 표본 표준편차, 괄호는 min~max. 검색은 결정적이므로 변동은 LLM 판정에서 옵니다.",
        "",
        "| variant | n | accuracy mean±sd (min~max) | not-gr recall mean±sd "
        "| not-gr F1 mean±sd | groundable 오차단 mean±sd | 판정 불안정 케이스 (flip rate, ids) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def ms(s: dict) -> str:
        if s["mean"] is None:
            return "-"
        sd = "-" if s["stdev"] is None else _fmt(s["stdev"])
        return f"{_fmt(s['mean'])}±{sd}"

    for name, v in stats["variants"].items():
        m = v["metrics"]
        acc = m["accuracy"]
        acc_cell = ms(acc) + (f" ({_fmt(acc['min'])}~{_fmt(acc['max'])})" if acc["mean"] is not None else "")
        ids = ", ".join(v["unstable_case_ids"]) if v["unstable_case_ids"] else "없음"
        lines.append(
            f"| {name} | {v['n']} | {acc_cell} | {ms(m['recall'])} | {ms(m['f1'])} | "
            f"{ms(m['groundable_false_block_rate'])} | "
            f"{len(v['unstable_case_ids'])}/{v['n']} ({_fmt(v['flip_rate'])}): {ids} |"
        )
    return "\n".join(lines).rstrip() + "\n"


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if pct else f"{value:.1f}"


def render_sweep_markdown(records: list[dict], thresholds: list[float] | None = None) -> str:
    """임계치 스윕 결과를 markdown 표로 렌더링합니다(순수 함수, API 호출 없음)."""
    thresholds = thresholds if thresholds is not None else SWEEP_THRESHOLDS
    lines = [
        "# 그라운딩 임계치 스윕 결과",
        "",
        f"케이스 수: {len(records)}",
        "",
        "| threshold | accuracy | not-gr precision | not-gr recall | not-gr F1 "
        "| filter false-block | LLM 호출 절감률 | LLM 단독 일치율 | 파싱 실패율 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for threshold in thresholds:
        m = compute_metrics(records, threshold)
        ng = m["not_groundable"]
        lines.append(
            f"| {threshold:.2f} | {_fmt(m['accuracy'])} | {_fmt(ng['precision'])} | "
            f"{_fmt(ng['recall'])} | {_fmt(ng['f1'])} | "
            f"{_fmt(m['filter_false_block_rate']['overall'])} | "
            f"{_fmt(m['llm_call_saving_rate'])} | {_fmt(m['llm_only_agreement'])} | "
            f"{_fmt(m['parse_failure_rate'])} |"
        )

    categories = sorted({str(r.get("category", "unknown")) for r in records})
    if categories:
        lines.append("")
        lines.append("## 카테고리별 filter false-block rate (라벨 groundable인데 유사도 미달)")
        lines.append("")
        lines.append("| threshold | " + " | ".join(categories) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in categories) + " |")
        for threshold in thresholds:
            m = compute_metrics(records, threshold)
            by_cat = m["filter_false_block_rate"]["by_category"]
            row = " | ".join(_fmt(by_cat.get(cat)) for cat in categories)
            lines.append(f"| {threshold:.2f} | {row} |")

    return "\n".join(lines).rstrip() + "\n"


def render_report(records: list[dict], threshold: float | None = None) -> str:
    """임계치 스윕(baseline 또는 유일한 variant)과, variant가 여러 개면 변형 비교 표를 합쳐 렌더링합니다.

    캐시에 repeat가 2개 이상이면 기존 섹션은 케이스 id 단위 집계라 중복 집계되지 않도록
    repeat 0 레코드만 사용하고, 마지막에 반복 측정 섹션을 덧붙입니다.
    """
    repeat_groups = group_by_repeat(records)
    multi_repeat = len(repeat_groups) > 1
    base_records = repeat_groups[min(repeat_groups)] if multi_repeat else records

    if multi_repeat and threshold is None:
        from app.config import get_settings

        threshold = get_settings().grounding_similarity_threshold

    grouped = group_by_variant(base_records)
    if len(grouped) <= 1:
        text = render_sweep_markdown(base_records)
    else:
        if threshold is None:
            from app.config import get_settings

            threshold = get_settings().grounding_similarity_threshold
        sweep_records = grouped.get("baseline") or next(iter(grouped.values()))
        text = render_sweep_markdown(sweep_records) + "\n" + render_variant_markdown(grouped, threshold)

    if not multi_repeat:
        return text
    note = (
        f"> 반복 {len(repeat_groups)}회 캐시: 위 섹션은 repeat 0 레코드만 사용합니다"
        "(케이스 id 단위 집계). 전체 repeat 통계는 아래 반복 측정 섹션을 보세요.\n"
    )
    return text + "\n" + note + "\n" + render_repeat_markdown(records, threshold)


# ---------------------------------------------------------------------------
# 라이브 수집 (실 GMS API 호출) — CLI에서만 사용, 순수 함수와 분리
# ---------------------------------------------------------------------------

async def _collect_records(
    dataset_path: Path,
    limit: int | None,
    variant_names: list[str] | None = None,
    repeat: int = 1,
) -> list[dict]:
    """tmp 워크스페이스를 시드하고 케이스마다 retrieve + (임계치 무관) call_grounding을
    호출해 결과 레코드를 만듭니다. RUN_LIVE_LLM 게이트는 호출부(main)의 책임입니다.

    repeat > 1이면 케이스당 검색은 1회만(결정적) 하고, 변형마다 판정을 repeat번 호출해
    LLM 비결정성에 따른 재현성을 잰다(레코드의 "repeat" 필드 = 0-based 반복 인덱스).
    LLM 호출 수 = 케이스 수 x 변형 수 x repeat.

    tests/live/conftest.py::live_env/seeded_workspace와 동일한 패턴(tmp
    VECTOR_STORE_PATH/METRICS_DIR + get_settings/get_vector_store 캐시 초기화)을
    사용합니다.
    """
    import os

    from app.config import get_settings
    from app.core.llm import provider as llm
    from app.core.rag import retriever
    from app.core.rag.grounding import build_grounding_prompt
    from app.core.rag.grounding_prompts import get_grounding_prompt
    from app.db.vector_store import get_vector_store
    from scripts.eval.seed import seed_workspace_async

    cases = load_dataset(dataset_path)
    if limit is not None:
        cases = cases[:limit]
    if variant_names is None:
        variant_names = list(VARIANTS)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        vector_store_path = tmp_path / "vector_store"
        metrics_dir = tmp_path / "metrics"
        vector_store_path.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        os.environ["VECTOR_STORE_PATH"] = str(vector_store_path)
        os.environ["METRICS_DIR"] = str(metrics_dir)
        get_settings.cache_clear()
        get_vector_store.cache_clear()

        db, workspace_id = await seed_workspace_async(tmp_path)
        try:
            records: list[dict] = []
            for case in cases:
                # 참고: history가 있는 케이스라도 query_rewriter는 호출하지 않고 question
                # 원문을 그대로 검색/판정에 사용합니다. 이는 app/core/rag/grounding.py::
                # assess()의 실제 입력(request.content, 항상 원문)과 동일하며, retrieve()에
                # rewritten_query를 쓰는 부분(A5 범위)은 이 스크립트에서 재현하지 않습니다.
                question = case["question"]
                chunks = await retriever.retrieve(question, workspace_id, db, top_k=5)
                top_similarity = max((c.similarity_score for c in chunks), default=0.0)

                for variant_name in variant_names:
                    cfg = VARIANTS[variant_name]
                    prompt = build_grounding_prompt(
                        chunks,
                        question,
                        top_k=cfg["top_k"],
                        max_chunk_chars=cfg["max_chunk_chars"],
                    )

                    for repeat_idx in range(repeat):
                        start = time.perf_counter()
                        llm_is_groundable = None
                        llm_confidence = None
                        prompt_tokens = None
                        completion_tokens = None
                        parse_ok = True
                        fallback_reason = None
                        error_type = None
                        try:
                            raw, usage = await llm.call_grounding(
                                prompt, system_prompt=get_grounding_prompt(cfg["prompt_version"])
                            )
                        except Exception as exc:
                            parse_ok = False
                            fallback_reason = "llm_error"
                            error_type = f"{type(exc).__name__}: {exc}"[:200]
                            print(f"[{variant_name}/{case['id']}] call_grounding 실패: {error_type}", file=sys.stderr)
                        else:
                            prompt_tokens = usage.prompt_tokens
                            completion_tokens = usage.completion_tokens
                            is_g = raw.get("is_groundable")
                            conf = raw.get("confidence")
                            if is_g is None or conf is None:
                                parse_ok = False
                                fallback_reason = "parse_error"
                            else:
                                llm_is_groundable = bool(is_g)
                                llm_confidence = float(conf)
                        latency_ms = (time.perf_counter() - start) * 1000

                        records.append(
                            {
                                "variant": variant_name,
                                "repeat": repeat_idx,
                                "id": case["id"],
                                "category": case["category"],
                                "expected_groundable": case["expected_groundable"],
                                "top_similarity": top_similarity,
                                "llm_is_groundable": llm_is_groundable,
                                "llm_confidence": llm_confidence,
                                "parse_ok": parse_ok,
                                "fallback_reason": fallback_reason,
                                "latency_ms": latency_ms,
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "prompt_chars": len(prompt),
                                "error_type": error_type,
                            }
                        )
            return records
        finally:
            db.close()
            get_settings.cache_clear()
            get_vector_store.cache_clear()


def _live_enabled() -> bool:
    """RUN_LIVE_LLM=1 AND settings.gms_api_key 비어있지 않음을 확인합니다(tests/live/conftest.py와 동일 게이트).

    .env가 자동 로드되므로 GMS_API_KEY가 이미 있어도 RUN_LIVE_LLM=1을 명시해야 과금이 발생합니다.
    """
    import os

    if os.environ.get("RUN_LIVE_LLM") != "1":
        return False

    from app.config import get_settings

    return bool(get_settings().gms_api_key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "그라운딩 임계치 오프라인 평가(A4). --live로 실 API 호출 결과를 캐시에 "
            "저장하거나, --from-cache로 저장된 캐시를 임계치 스윕합니다(API 호출 없음)."
        )
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="실 GMS API를 호출해 데이터셋 전 케이스를 수집하고 결과를 data/eval/에 캐시로 저장합니다. "
        "RUN_LIVE_LLM=1 환경변수와 비어있지 않은 GMS_API_KEY가 필요합니다(비용 발생).",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="--live로 생성된 캐시 jsonl 경로. API 호출 없이 임계치 0.20~0.60을 스윕해 markdown 표를 출력합니다.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"평가 케이스 jsonl 경로 (기본: {_DEFAULT_DATASET.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="--live 수집 시 처리할 케이스 수를 제한합니다(스모크 실행용, 비용 절감).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="--live 결과 캐시 저장 경로 (기본: data/eval/grounding-{YYYYMMDD-HHMMSS}.jsonl)",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=None,
        help="--live 수집 시 쉼표로 구분한 판정 변형 목록 (기본 전체: "
        + ", ".join(VARIANTS)
        + "). 케이스당 검색은 1회, 변형마다 판정 호출 1회가 발생합니다.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="변형 비교 표에 적용할 유사도 임계치 (기본: settings.grounding_similarity_threshold)",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="--live 전용. 변형마다 판정을 N번 반복해 LLM 비결정성(재현성)을 잽니다(N>=1, 기본 1). "
        "검색은 케이스당 1회. LLM 호출 수 = 케이스 수 x 변형 수 x N (비용 N배).",
    )
    args = parser.parse_args(argv)

    if args.repeat < 1:
        parser.error("--repeat는 1 이상이어야 합니다.")
    if args.repeat != 1 and not args.live:
        parser.error("--repeat는 --live와 함께만 사용할 수 있습니다.")

    if args.live and args.from_cache:
        parser.error("--live와 --from-cache는 동시에 지정할 수 없습니다.")

    if args.from_cache:
        records = load_cache(args.from_cache)
        print(render_report(records, args.threshold))
        return 0

    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()] if args.variants else None
    unknown = [v for v in (variant_names or []) if v not in VARIANTS]
    if unknown:
        parser.error(f"알 수 없는 variant: {', '.join(unknown)} (사용 가능: {', '.join(VARIANTS)})")

    if args.live:
        if not _live_enabled():
            print(
                "RUN_LIVE_LLM=1 과 비어있지 않은 GMS_API_KEY가 필요합니다 "
                "(docs/grounding-eval.md 참고). 비용이 발생하는 실 API 호출입니다.",
                file=sys.stderr,
            )
            return 1

        records = asyncio.run(_collect_records(args.dataset, args.limit, variant_names, args.repeat))

        out_path = args.out
        if out_path is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = _DEFAULT_CACHE_DIR / f"grounding-{timestamp}.jsonl"
        save_cache(out_path, records)
        print(f"수집 완료: {len(records)}건 -> {out_path}")
        print()
        print(render_report(records, args.threshold))
        return 0

    parser.error("--live 또는 --from-cache 중 하나를 지정하세요.")
    return 2  # pragma: no cover — argparse.error()가 SystemExit을 던지므로 도달하지 않음


if __name__ == "__main__":
    raise SystemExit(main())

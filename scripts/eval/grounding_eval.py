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

캐시 레코드 스키마(jsonl 1줄 = 케이스 1건):
    {id, category, expected_groundable, top_similarity, llm_is_groundable,
     llm_confidence, parse_ok, fallback_reason, latency_ms, prompt_tokens,
     completion_tokens}

fail-open 반영: app/core/rag/grounding.py::assess()는 call_grounding 예외
("llm_error") 또는 파싱 실패("parse_error") 시 is_groundable=True로 fail-open
합니다. 이 스크립트는 임계치와 무관하게 항상 call_grounding을 호출하므로, 위 두
경우에도 llm_is_groundable/llm_confidence를 기록하지 못합니다(parse_ok=False,
fallback_reason만 남음). compute_metrics()는 이 fallback 레코드를 실제 앱과 동일하게
"LLM 단계까지 도달했다면 항상 groundable=True"로 취급합니다.
"""

import argparse
import json
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "scripts" / "eval" / "datasets" / "grounding_cases.jsonl"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "eval"

# 오프라인 스윕 임계치: 0.20 ~ 0.60, 0.05 간격 (부동소수 오차 방지를 위해 정수 스텝으로 계산)
SWEEP_THRESHOLDS = [round(0.20 + 0.05 * i, 2) for i in range(9)]

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


# ---------------------------------------------------------------------------
# CLI (--from-cache만 우선 제공. --live 라이브 수집은 뒤이은 커밋에서 추가됩니다.)
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "그라운딩 임계치 오프라인 평가(A4). --from-cache로 저장된 캐시를 "
            "임계치 스윕합니다(API 호출 없음)."
        )
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
    args = parser.parse_args(argv)

    if args.from_cache:
        records = load_cache(args.from_cache)
        print(render_sweep_markdown(records))
        return 0

    parser.error("--from-cache를 지정하세요(라이브 수집 --live는 다음 커밋에서 추가됩니다).")
    return 2  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())

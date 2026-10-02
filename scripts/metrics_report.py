"""
관측성 메트릭 JSONL 집계 스크립트 (R-4, C-4, A2).

app.core.metrics.record_metric()이 남긴 {metrics_dir}/{event}.jsonl 파일들을 읽어
이벤트(파일)별로 파싱 성공률, 단계별(*_ms) p50/p95, 실패 사유 분포를 markdown으로
출력합니다. API 키가 없어도 로컬 JSONL만으로 동작합니다.

사용법:
    python3 scripts/metrics_report.py <jsonl_dir>

집계 정의:
- 파싱 성공률: result 필드가 있는 레코드 중 "FAILED"가 아닌 비율(%). PARSE_WARN/EMPTY는
  인덱싱 자체는 계속 진행되었으므로 성공으로 집계하되, 실패 사유 분포에는 포함하지 않음.
- 단계별 p50/p95: 키 이름이 "_ms"로 끝나는 필드 중 숫자(None 제외) 값의 분위수.
- 실패 사유 분포: result == "FAILED"인 레코드의 failure_stage / error_type 값별 개수.
- llm_calls 전용: purpose×model별 호출 수, latency_ms p50/p95, 평균 prompt/completion
  토큰, 파싱 실패율(parse_ok=False 비율), 에러 수(error_type 존재 레코드 수)를 별도
  표로 집계 (app/core/llm/provider.py의 LLM 호출 단위 관측성).
"""

import argparse
import json
import math
import sys
from pathlib import Path


def load_records(jsonl_dir: Path) -> dict[str, list[dict]]:
    """jsonl_dir 하위의 *.jsonl 파일을 이벤트(파일명, 확장자 제외)별로 읽어 반환합니다.

    JSON 파싱에 실패한 줄은 건너뜁니다(다른 프로세스가 쓰는 도중일 수 있음).
    """
    records_by_event: dict[str, list[dict]] = {}

    for path in sorted(jsonl_dir.glob("*.jsonl")):
        event = path.stem
        records: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        records_by_event[event] = records

    return records_by_event


def _percentile(values: list[float], pct: float) -> float:
    """values의 pct(0~100) 백분위수를 선형 보간으로 계산합니다."""
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]

    lower_value = ordered[int(lower)] * (upper - rank)
    upper_value = ordered[int(upper)] * (rank - lower)
    return lower_value + upper_value


def compute_stage_percentiles(records: list[dict]) -> dict[str, tuple[float, float]]:
    """"_ms"로 끝나는 필드별 (p50, p95)를 계산합니다. 값이 없는 필드는 제외합니다."""
    values_by_stage: dict[str, list[float]] = {}

    for record in records:
        for key, value in record.items():
            if not key.endswith("_ms"):
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            values_by_stage.setdefault(key, []).append(float(value))

    return {
        stage: (_percentile(values, 50), _percentile(values, 95))
        for stage, values in sorted(values_by_stage.items())
    }


def compute_result_summary(records: list[dict]) -> dict[str, object] | None:
    """result 필드 분포와 성공률(비-FAILED 비율)을 계산합니다. result 필드가 없으면 None."""
    with_result = [r for r in records if "result" in r]
    if not with_result:
        return None

    counts: dict[str, int] = {}
    for record in with_result:
        result = str(record["result"])
        counts[result] = counts.get(result, 0) + 1

    total = len(with_result)
    failed = counts.get("FAILED", 0)
    success_rate = (total - failed) / total * 100

    return {"total": total, "counts": counts, "success_rate": success_rate}


def compute_failure_distribution(records: list[dict]) -> dict[str, dict[str, int]]:
    """result == "FAILED" 레코드의 failure_stage / error_type 분포를 계산합니다."""
    failed = [r for r in records if r.get("result") == "FAILED"]

    stage_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}

    for record in failed:
        stage = str(record.get("failure_stage") or "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
        error = str(record.get("error_type") or "unknown")
        error_counts[error] = error_counts.get(error, 0) + 1

    return {"failure_stage": stage_counts, "error_type": error_counts}


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compute_llm_calls_breakdown(records: list[dict]) -> dict[tuple[str, str], dict[str, object]]:
    """llm_calls 이벤트를 (purpose, model)별로 묶어 호출 수/지연/토큰/파싱 실패율/에러 수를 계산합니다.

    purpose/model 키가 없는 레코드는 "unknown"으로 묶습니다.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        purpose = str(record.get("purpose") or "unknown")
        model = str(record.get("model") or "unknown")
        groups.setdefault((purpose, model), []).append(record)

    breakdown: dict[tuple[str, str], dict[str, object]] = {}
    for key, group in groups.items():
        latencies = [float(r["latency_ms"]) for r in group if _is_number(r.get("latency_ms"))]
        prompt_tokens = [float(r["prompt_tokens"]) for r in group if _is_number(r.get("prompt_tokens"))]
        completion_tokens = [
            float(r["completion_tokens"]) for r in group if _is_number(r.get("completion_tokens"))
        ]
        parse_ok_values = [r["parse_ok"] for r in group if "parse_ok" in r]
        parse_failures = sum(1 for v in parse_ok_values if v is False)
        error_count = sum(1 for r in group if r.get("error_type"))

        breakdown[key] = {
            "count": len(group),
            "p50_latency_ms": _percentile(latencies, 50),
            "p95_latency_ms": _percentile(latencies, 95),
            "avg_prompt_tokens": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0.0,
            "avg_completion_tokens": (
                sum(completion_tokens) / len(completion_tokens) if completion_tokens else 0.0
            ),
            "parse_failure_rate": (
                parse_failures / len(parse_ok_values) * 100 if parse_ok_values else 0.0
            ),
            "error_count": error_count,
        }

    return breakdown


def compute_cache_summary(records: list[dict]) -> dict[str, object] | None:
    """chat_metrics 레코드에서 시맨틱 캐시 hit율과 hit/miss별 지연 백분위수를 계산합니다.

    cache_enabled가 True인 레코드가 하나도 없으면 None을 반환합니다.
    """
    enabled = [r for r in records if r.get("cache_enabled") is True]
    if not enabled:
        return None

    hits = [r for r in enabled if r.get("cache_hit") is True]
    misses = [r for r in enabled if r.get("cache_hit") is not True]

    def percentiles(group: list[dict], key: str) -> tuple[float, float] | None:
        values = [r[key] for r in group if _is_number(r.get(key))]
        if not values:
            return None
        return _percentile(values, 50), _percentile(values, 95)

    # LLM 재검증(cache_verify_*) 지표. 구버전 레코드는 필드가 없으므로 None으로 취급합니다.
    verified = [r for r in enabled if r.get("cache_verify_result") is not None]
    verify_outcomes: dict[str, int] = {}
    for r in verified:
        outcome = str(r["cache_verify_result"])
        verify_outcomes[outcome] = verify_outcomes.get(outcome, 0) + 1
    verify_hit_count = sum(1 for r in hits if r.get("cache_verify_result") == "yes")

    return {
        "enabled_count": len(enabled),
        "hit_count": len(hits),
        "hit_rate": len(hits) / len(enabled) * 100,
        "verify_count": len(verified),
        "verify_outcomes": verify_outcomes,
        "verify_latency": percentiles(verified, "cache_verify_ms"),
        "verify_hit_count": verify_hit_count,
        "verify_hit_share": (verify_hit_count / len(hits) * 100) if hits else 0.0,
        "latency": {
            (label, key): percentiles(group, key)
            for label, group in (("hit", hits), ("miss", misses))
            for key in ("total_ms", "ttft_ms")
        },
    }


def render_markdown(records_by_event: dict[str, list[dict]]) -> str:
    """이벤트별 집계 결과를 markdown 문자열로 렌더링합니다."""
    lines: list[str] = ["# 메트릭 집계 리포트", ""]

    if not records_by_event:
        lines.append("(집계할 .jsonl 파일이 없습니다)")
        return "\n".join(lines)

    for event, records in records_by_event.items():
        lines.append(f"## {event} ({len(records)}건)")
        lines.append("")

        summary = compute_result_summary(records)
        if summary is not None:
            lines.append(f"- 성공률(비-FAILED 비율): {summary['success_rate']:.1f}%")
            counts_text = ", ".join(f"{k}={v}" for k, v in sorted(summary["counts"].items()))
            lines.append(f"- result 분포: {counts_text}")
            lines.append("")

        stage_percentiles = compute_stage_percentiles(records)
        if stage_percentiles:
            lines.append("| 구간 | p50(ms) | p95(ms) |")
            lines.append("| --- | --- | --- |")
            for stage, (p50, p95) in stage_percentiles.items():
                lines.append(f"| {stage} | {p50:.1f} | {p95:.1f} |")
            lines.append("")

        failure_dist = compute_failure_distribution(records)
        if failure_dist["failure_stage"] or failure_dist["error_type"]:
            lines.append("- 실패 단계 분포: " + ", ".join(
                f"{k}={v}" for k, v in sorted(failure_dist["failure_stage"].items())
            ))
            lines.append("- 실패 원인(예외 클래스) 분포: " + ", ".join(
                f"{k}={v}" for k, v in sorted(failure_dist["error_type"].items())
            ))
            lines.append("")

        if event == "chat_metrics":
            cache_summary = compute_cache_summary(records)
            if cache_summary is None:
                lines.append("- 시맨틱 캐시 hit율: n/a (cache_enabled 레코드 없음)")
            else:
                lines.append(
                    f"- 시맨틱 캐시 hit율: {cache_summary['hit_rate']:.1f}% "
                    f"({cache_summary['hit_count']}/{cache_summary['enabled_count']})"
                )
                if cache_summary["verify_count"] == 0:
                    lines.append("- 재검증: n/a")
                else:
                    outcomes = cache_summary["verify_outcomes"]
                    outcome_text = " / ".join(
                        f"{name} {outcomes.get(name, 0)}"
                        for name in ("yes", "no", "invalid", "error", "timeout")
                    )
                    lines.append(f"- 재검증 호출: {cache_summary['verify_count']} ({outcome_text})")
                    vlat = cache_summary["verify_latency"]
                    if vlat is None:
                        lines.append("- 재검증 지연: p50 - / p95 -")
                    else:
                        lines.append(f"- 재검증 지연: p50 {vlat[0]:.1f}ms / p95 {vlat[1]:.1f}ms")
                    lines.append(
                        f"- 재검증 경유 hit 비율: {cache_summary['verify_hit_share']:.1f}% "
                        f"({cache_summary['verify_hit_count']}/{cache_summary['hit_count']})"
                    )
                lines.append("")
                lines.append("| 구분 | 지표 | p50(ms) | p95(ms) |")
                lines.append("| --- | --- | --- | --- |")
                for (label, key), pcts in cache_summary["latency"].items():
                    if pcts is None:
                        lines.append(f"| cache_hit={label == 'hit'} | {key} | - | - |")
                    else:
                        lines.append(
                            f"| cache_hit={label == 'hit'} | {key} | {pcts[0]:.1f} | {pcts[1]:.1f} |"
                        )
            lines.append("")

        if event == "llm_calls":
            breakdown = compute_llm_calls_breakdown(records)
            if breakdown:
                lines.append(
                    "| purpose | model | 호출 수 | p50(ms) | p95(ms) | "
                    "평균 prompt_tokens | 평균 completion_tokens | 파싱 실패율(%) | 에러 수 |"
                )
                lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
                for (purpose, model), stats in sorted(breakdown.items()):
                    lines.append(
                        f"| {purpose} | {model} | {stats['count']} | "
                        f"{stats['p50_latency_ms']:.1f} | {stats['p95_latency_ms']:.1f} | "
                        f"{stats['avg_prompt_tokens']:.1f} | {stats['avg_completion_tokens']:.1f} | "
                        f"{stats['parse_failure_rate']:.1f} | {stats['error_count']} |"
                    )
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="metrics JSONL 집계 리포트 생성")
    parser.add_argument("jsonl_dir", help="record_metric()이 기록한 {event}.jsonl 파일들이 있는 디렉터리")
    args = parser.parse_args(argv)

    jsonl_dir = Path(args.jsonl_dir)
    if not jsonl_dir.is_dir():
        print(f"디렉터리를 찾을 수 없습니다: {jsonl_dir}", file=sys.stderr)
        return 1

    records_by_event = load_records(jsonl_dir)
    print(render_markdown(records_by_event))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

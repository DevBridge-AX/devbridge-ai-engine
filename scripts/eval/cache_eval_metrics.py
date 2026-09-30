"""
시맨틱 캐시 임계치 평가용 순수 지표 함수 (A9).

이 모듈은 API 호출이나 DB/벡터스토어 접근을 하지 않는 순수 함수만 포함합니다.
`scripts/eval/cache_eval.py`가 질문 쌍의 임베딩 코사인 유사도를 수집해 캐시 jsonl로
저장하면, 이 모듈이 임계치 스윕/권고/분포/markdown 렌더링을 담당합니다.
`tests/test_cache_eval_metrics.py`가 합성 레코드로 단위 테스트합니다(실 API 불필요).

레코드 형태: {id, kind, q1, q2, similarity} (kind: same_intent | different_intent)
- same_intent: 같은 질문의 다른 표현 — 캐시 hit이 정답
- different_intent: 표면은 비슷하지만 답이 다른 질문 — hit이면 오답(false hit)
"""

import json
import math
from pathlib import Path

KIND_SAME = "same_intent"
KIND_DIFFERENT = "different_intent"
VALID_KINDS = (KIND_SAME, KIND_DIFFERENT)

# 오프라인 스윕 임계치: 0.85 ~ 0.99, 0.01 간격 (부동소수 오차 방지를 위해 정수 스텝으로 계산)
SWEEP_THRESHOLDS = [round(0.85 + 0.01 * i, 2) for i in range(15)]

_REQUIRED_FIELDS = ("id", "kind", "q1", "q2")


# ---------------------------------------------------------------------------
# 유사도 / I/O
# ---------------------------------------------------------------------------

def cosine_similarity(a: list[float], b: list[float]) -> float:
    """두 벡터의 코사인 유사도를 계산합니다. 영벡터가 있으면 0.0을 반환합니다."""
    if len(a) != len(b):
        raise ValueError("cosine_similarity: 벡터 길이가 다릅니다.")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def load_dataset(path: Path) -> list[dict]:
    """scripts/eval/datasets/cache_pairs.jsonl 형식의 질문 쌍을 읽고 필드를 검증합니다.

    id/kind/q1/q2가 없거나 kind가 same_intent/different_intent가 아니면 ValueError.
    """
    pairs = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        pair = json.loads(line)
        for field in _REQUIRED_FIELDS:
            if not pair.get(field):
                raise ValueError(f"{path}:{lineno}: 필수 필드 '{field}'가 없거나 비어 있습니다.")
        if pair["kind"] not in VALID_KINDS:
            raise ValueError(
                f"{path}:{lineno}: kind는 {VALID_KINDS} 중 하나여야 합니다 (got {pair['kind']!r})."
            )
        pairs.append(pair)
    return pairs


def load_cache(path: Path) -> list[dict]:
    """cache_eval.py --collect로 생성된 결과 캐시(jsonl)를 읽습니다. 파싱 실패한 줄은 건너뜁니다."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def save_cache(path: Path, records: list[dict]) -> None:
    """수집된 레코드를 jsonl로 저장합니다(1줄 = 질문 쌍 1건)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")


# ---------------------------------------------------------------------------
# 집계
# ---------------------------------------------------------------------------

def sweep(records: list[dict], thresholds: list[float] | None = None) -> list[dict]:
    """임계치별 캐시 hit 품질 지표를 계산합니다.

    hit 판정: similarity >= threshold.
    - hit_rate(= recall): same_intent 중 hit 비율 (캐시 절감 효과)
    - false_hit_rate: different_intent 중 hit 비율 (오답 위험)
    - precision: 전체 hit 중 same_intent 비율 (hit가 없으면 None)
    - f1: precision/recall 조화평균 (계산 불가 시 None)
    """
    thresholds = thresholds if thresholds is not None else SWEEP_THRESHOLDS
    same = [float(r["similarity"]) for r in records if r["kind"] == KIND_SAME]
    diff = [float(r["similarity"]) for r in records if r["kind"] == KIND_DIFFERENT]

    rows = []
    for t in thresholds:
        same_hits = sum(1 for s in same if s >= t)
        diff_hits = sum(1 for s in diff if s >= t)
        total_hits = same_hits + diff_hits

        hit_rate = same_hits / len(same) if same else None
        false_hit_rate = diff_hits / len(diff) if diff else None
        precision = same_hits / total_hits if total_hits else None
        f1 = (
            2 * precision * hit_rate / (precision + hit_rate)
            if precision is not None and hit_rate is not None and (precision + hit_rate) > 0
            else None
        )
        rows.append(
            {
                "threshold": t,
                "hit_rate": hit_rate,
                "false_hit_rate": false_hit_rate,
                "precision": precision,
                "recall": hit_rate,
                "f1": f1,
                "same_hits": same_hits,
                "same_total": len(same),
                "diff_hits": diff_hits,
                "diff_total": len(diff),
            }
        )
    return rows


def recommend_threshold(sweep_rows: list[dict], max_false_hit_rate: float = 0.0) -> dict | None:
    """false_hit_rate <= max_false_hit_rate 인 행 중 hit_rate가 가장 높은 행을 고릅니다.

    동률이면 가장 낮은 임계치를 선택합니다. 조건을 만족하는 행이 없으면 None.
    """
    eligible = [
        r
        for r in sweep_rows
        if r["false_hit_rate"] is not None
        and r["hit_rate"] is not None
        and r["false_hit_rate"] <= max_false_hit_rate
    ]
    if not eligible:
        return None
    return min(eligible, key=lambda r: (-r["hit_rate"], r["threshold"]))


def _percentile(values: list[float], pct: float) -> float:
    """values의 pct(0~100) 백분위수를 선형 보간으로 계산합니다 (rewrite_metrics와 동일 방식)."""
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


def similarity_distribution(records: list[dict]) -> dict[str, dict]:
    """kind별 유사도 분포 {min, p25, median, p75, max, mean}을 계산합니다(레코드가 없는 kind는 생략)."""
    result: dict[str, dict] = {}
    for kind in VALID_KINDS:
        values = [float(r["similarity"]) for r in records if r["kind"] == kind]
        if not values:
            continue
        result[kind] = {
            "min": min(values),
            "p25": _percentile(values, 25),
            "median": _percentile(values, 50),
            "p75": _percentile(values, 75),
            "max": max(values),
            "mean": sum(values) / len(values),
        }
    return result


# ---------------------------------------------------------------------------
# markdown 렌더링
# ---------------------------------------------------------------------------

def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def _rec_line(label: str, rec: dict | None) -> str:
    if rec is None:
        return f"- {label}: 조건을 만족하는 임계치 없음 (스윕 범위 내에서 false hit를 허용 한도 이하로 낮출 수 없음)"
    return (
        f"- {label}: **{rec['threshold']:.2f}** "
        f"(hit_rate {_pct(rec['hit_rate'])}, false_hit_rate {_pct(rec['false_hit_rate'])}, "
        f"hits {rec['same_hits']}/{rec['same_total']}, false hits {rec['diff_hits']}/{rec['diff_total']})"
    )


def render_markdown(
    records: list[dict],
    sweep_rows: list[dict],
    recommendations: dict[str, dict | None],
    embedding_model: str = "-",
) -> str:
    """스윕 결과를 markdown으로 렌더링합니다(순수 함수).

    recommendations: {라벨: recommend_threshold 결과}. 예) {"false hit 0%": ..., "false hit <= 2%": ...}
    """
    n_same = sum(1 for r in records if r["kind"] == KIND_SAME)
    n_diff = sum(1 for r in records if r["kind"] == KIND_DIFFERENT)

    lines = [
        "# 시맨틱 캐시 임계치 스윕 결과",
        "",
        "## 실험 조건",
        "",
        f"- 임베딩 모델: {embedding_model}",
        f"- same_intent 쌍: {n_same}건",
        f"- different_intent 쌍: {n_diff}건",
        "- hit 판정: 코사인 유사도 >= 임계치",
        "",
        "## 유사도 분포",
        "",
        "| kind | min | p25 | median | p75 | max | mean |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for kind, d in similarity_distribution(records).items():
        lines.append(
            f"| {kind} | {d['min']:.4f} | {d['p25']:.4f} | {d['median']:.4f} | "
            f"{d['p75']:.4f} | {d['max']:.4f} | {d['mean']:.4f} |"
        )

    lines += [
        "",
        "## 임계치 스윕",
        "",
        "| threshold | hit_rate | false_hit_rate | precision | f1 | hits/same | hits/diff |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in sweep_rows:
        f1 = "-" if r["f1"] is None else f"{r['f1']:.3f}"
        lines.append(
            f"| {r['threshold']:.2f} | {_pct(r['hit_rate'])} | {_pct(r['false_hit_rate'])} | "
            f"{_pct(r['precision'])} | {f1} | "
            f"{r['same_hits']}/{r['same_total']} | {r['diff_hits']}/{r['diff_total']} |"
        )

    lines += ["", "## 권고 임계치", ""]
    for label, rec in recommendations.items():
        lines.append(_rec_line(label, rec))

    lines += ["", "## 경계 사례", "", "### 유사도가 가장 낮은 same_intent 5건 (hit를 놓치기 쉬운 쌍)", ""]
    lines += _boundary_table(
        sorted((r for r in records if r["kind"] == KIND_SAME), key=lambda r: r["similarity"])[:5]
    )
    lines += ["", "### 유사도가 가장 높은 different_intent 5건 (false hit 위험 쌍)", ""]
    lines += _boundary_table(
        sorted(
            (r for r in records if r["kind"] == KIND_DIFFERENT),
            key=lambda r: r["similarity"],
            reverse=True,
        )[:5]
    )

    return "\n".join(lines).rstrip() + "\n"


def _boundary_table(rows: list[dict]) -> list[str]:
    lines = ["| id | similarity | q1 | q2 |", "| --- | --- | --- | --- |"]
    for r in rows:
        lines.append(f"| {r['id']} | {float(r['similarity']):.4f} | {r['q1']} | {r['q2']} |")
    return lines

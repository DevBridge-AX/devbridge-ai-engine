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

# LLM 재검증(verify) 평가: 후보 하한 스윕값과 즉시 hit 임계치(운영 기본 semantic_cache_threshold)
VERIFY_CANDIDATE_THRESHOLDS = [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]
DEFAULT_DIRECT_THRESHOLD = 0.95
# 채택 기준: false_hit_rate <= 2% AND hit_rate >= 50%
ADOPT_MAX_FALSE_HIT_RATE = 0.02
ADOPT_MIN_HIT_RATE = 0.50

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
    embedding_task_type: str | None = None,
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
        f"- 임베딩 모델: {embedding_model}"
        + (f" (taskType: {embedding_task_type})" if embedding_task_type else ""),
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


# ---------------------------------------------------------------------------
# LLM 재검증(verify) 평가 — verify_* 필드가 있는 캐시 레코드에서만 의미가 있음
# ---------------------------------------------------------------------------

def has_verify_fields(records: list[dict]) -> bool:
    """레코드 중 하나라도 verify 결과(verify_same)를 가지면 True. 구버전 캐시는 False."""
    return any(r.get("verify_same") is not None for r in records)


def verify_sweep(
    records: list[dict],
    candidate_thresholds: list[float] | None = None,
    direct_threshold: float = DEFAULT_DIRECT_THRESHOLD,
) -> list[dict]:
    """후보 하한(candidate_threshold)별 "임계치 + 재검증" hit 품질을 계산합니다.

    hit 판정(파이프라인과 동일):
    - similarity >= direct_threshold: 검증 없이 hit
    - candidate_threshold <= similarity < direct_threshold: verify_same이 True일 때만 hit
    - 그 미만: miss
    verify_same이 없는 레코드는 후보 구간에서 miss로 취급합니다.
    verify_calls: 후보 구간에 들어가 검증이 필요한 쌍 수(= 실제 운영에서 발생할 LLM 호출 수).
    adopt: false_hit_rate <= 2% AND hit_rate >= 50%.
    """
    thresholds = candidate_thresholds if candidate_thresholds is not None else VERIFY_CANDIDATE_THRESHOLDS
    same = [r for r in records if r["kind"] == KIND_SAME]
    diff = [r for r in records if r["kind"] == KIND_DIFFERENT]

    def _hit(record: dict, candidate: float) -> tuple[bool, bool]:
        """(hit 여부, 검증 호출 필요 여부)"""
        sim = float(record["similarity"])
        if sim >= direct_threshold:
            return True, False
        if sim >= candidate:
            return bool(record.get("verify_same")), True
        return False, False

    rows = []
    for t in thresholds:
        same_res = [_hit(r, t) for r in same]
        diff_res = [_hit(r, t) for r in diff]
        same_hits = sum(1 for hit, _ in same_res if hit)
        diff_hits = sum(1 for hit, _ in diff_res if hit)
        hit_rate = same_hits / len(same) if same else None
        false_hit_rate = diff_hits / len(diff) if diff else None
        adopt = (
            hit_rate is not None
            and false_hit_rate is not None
            and false_hit_rate <= ADOPT_MAX_FALSE_HIT_RATE
            and hit_rate >= ADOPT_MIN_HIT_RATE
        )
        rows.append(
            {
                "candidate_threshold": t,
                "direct_threshold": direct_threshold,
                "hit_rate": hit_rate,
                "false_hit_rate": false_hit_rate,
                "same_hits": same_hits,
                "same_total": len(same),
                "diff_hits": diff_hits,
                "diff_total": len(diff),
                "verify_calls": sum(1 for _, v in same_res + diff_res if v),
                "verify_calls_same": sum(1 for _, v in same_res if v),
                "verify_calls_diff": sum(1 for _, v in diff_res if v),
                "adopt": adopt,
            }
        )
    return rows


def verify_stats(records: list[dict]) -> dict:
    """검증기 단독 성능을 계산합니다(유사도와 무관하게 verify 결과가 있는 모든 쌍 대상).

    - accuracy: same_intent는 verify_same=True, different_intent는 False면 정답
    - misjudged_ids: 오판 쌍 id (false_yes: 다른 질문을 YES / false_no: 같은 질문을 NO)
    - latency p50/p95(ms), 호출당 평균 prompt/completion 토큰, outcome별 건수
    """
    verified = [r for r in records if r.get("verify_same") is not None]
    if not verified:
        return {"total": 0}

    false_yes = [r["id"] for r in verified if r["kind"] == KIND_DIFFERENT and r["verify_same"]]
    false_no = [r["id"] for r in verified if r["kind"] == KIND_SAME and not r["verify_same"]]
    latencies = [float(r["verify_latency_ms"]) for r in verified if r.get("verify_latency_ms") is not None]
    prompt_tokens = [r["verify_prompt_tokens"] for r in verified if r.get("verify_prompt_tokens") is not None]
    completion_tokens = [
        r["verify_completion_tokens"] for r in verified if r.get("verify_completion_tokens") is not None
    ]
    outcomes: dict[str, int] = {}
    for r in verified:
        key = r.get("verify_outcome") or "-"
        outcomes[key] = outcomes.get(key, 0) + 1

    total = len(verified)
    return {
        "total": total,
        "accuracy": (total - len(false_yes) - len(false_no)) / total,
        "false_yes_ids": false_yes,
        "false_no_ids": false_no,
        "misjudged_ids": false_yes + false_no,
        "latency_p50_ms": _percentile(latencies, 50) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 95) if latencies else None,
        "avg_prompt_tokens": sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None,
        "avg_completion_tokens": sum(completion_tokens) / len(completion_tokens) if completion_tokens else None,
        "outcomes": outcomes,
        "models": sorted({r["verify_model"] for r in verified if r.get("verify_model")}),
        "prompt_versions": sorted({r["verify_prompt_version"] for r in verified if r.get("verify_prompt_version")}),
    }


def _num(value: float | None, fmt: str = ".1f") -> str:
    return "-" if value is None else format(value, fmt)


def render_verify_markdown(
    records: list[dict],
    rows: list[dict],
    stats: dict,
    direct_threshold: float = DEFAULT_DIRECT_THRESHOLD,
) -> str:
    """LLM 재검증 스윕 결과를 markdown 섹션으로 렌더링합니다(순수 함수)."""
    lines = [
        "",
        "## LLM 재검증 스윕",
        "",
        f"- 즉시 hit 임계치(direct): {direct_threshold:.2f} (이상이면 검증 없이 hit)",
        "- 후보 구간: candidate <= 유사도 < direct → 검증 YES일 때만 hit",
        f"- 채택 기준: false_hit_rate <= {ADOPT_MAX_FALSE_HIT_RATE * 100:.0f}% AND "
        f"hit_rate >= {ADOPT_MIN_HIT_RATE * 100:.0f}%",
        f"- 검증 모델: {', '.join(stats.get('models', [])) or '-'} / "
        f"프롬프트 버전: {', '.join(stats.get('prompt_versions', [])) or '-'}",
        "",
        "| candidate | hit_rate | false_hit_rate | hits/same | hits/diff | verify 호출(same/diff) | 채택 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['candidate_threshold']:.2f} | {_pct(r['hit_rate'])} | {_pct(r['false_hit_rate'])} | "
            f"{r['same_hits']}/{r['same_total']} | {r['diff_hits']}/{r['diff_total']} | "
            f"{r['verify_calls']} ({r['verify_calls_same']}/{r['verify_calls_diff']}) | "
            f"{'O' if r['adopt'] else '-'} |"
        )

    lines += ["", "### 검증기 단독 성능 (유사도 무관, 전 쌍)", ""]
    if not stats.get("total"):
        lines.append("- verify 결과 없음")
    else:
        lines += [
            f"- 정확도: {_pct(stats['accuracy'])} ({stats['total']}쌍)",
            f"- 오판 id: {', '.join(stats['misjudged_ids']) or '없음'} "
            f"(false YES: {', '.join(stats['false_yes_ids']) or '없음'} / "
            f"false NO: {', '.join(stats['false_no_ids']) or '없음'})",
            f"- 지연 p50/p95: {_num(stats['latency_p50_ms'])} / {_num(stats['latency_p95_ms'])} ms "
            "(hit 경로 지연 추정 = 기존 hit 경로 p50 + verify p50)",
            f"- 호출당 평균 토큰: prompt {_num(stats['avg_prompt_tokens'])} / "
            f"completion {_num(stats['avg_completion_tokens'])}",
            "- outcome: " + ", ".join(f"{k} {v}" for k, v in sorted(stats["outcomes"].items())),
        ]
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# 검증 프롬프트 버전별 비교 (3회차)
# ---------------------------------------------------------------------------

# 비교표에 표시할 후보 하한(hit_rate@candidate)
COMPARE_CANDIDATES = [0.80, 0.85, 0.86]


def parse_prompt_versions(value: str | None, default: str) -> list[str]:
    """`--verify-prompt-version` 값("v2" 또는 "v1,v2")을 중복 없는 버전 목록으로 변환합니다.

    값이 없으면 [default]. 빈 항목은 무시하고, 목록이 비면 ValueError.
    """
    if value is None:
        return [default]
    versions: list[str] = []
    for part in value.split(","):
        part = part.strip()
        if part and part not in versions:
            versions.append(part)
    if not versions:
        raise ValueError("검증 프롬프트 버전이 비어 있습니다.")
    return versions


def group_by_prompt_version(records: list[dict]) -> dict[str, list[dict]]:
    """verify_prompt_version별로 레코드를 묶습니다(입력 순서 유지).

    필드가 없는 구버전 레코드는 키 ""(빈 문자열)로 묶입니다.
    """
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r.get("verify_prompt_version") or "", []).append(r)
    return groups


def has_multiple_prompt_versions(records: list[dict]) -> bool:
    """서로 다른 verify_prompt_version 값이 2개 이상이면 True(구버전 캐시/단일 버전은 False)."""
    return len({r.get("verify_prompt_version") for r in records if r.get("verify_prompt_version")}) > 1


def render_version_comparison(
    groups: dict[str, list[dict]],
    direct_threshold: float = DEFAULT_DIRECT_THRESHOLD,
    candidates: list[float] | None = None,
) -> str:
    """버전별 검증기 단독 성능 + hit_rate@candidate + false_hit_rate 비교표(markdown)."""
    cands = candidates if candidates is not None else COMPARE_CANDIDATES
    head = ["version", "verifier accuracy", "false YES", "false NO"]
    head += [f"hit_rate@{c:.2f}" for c in cands] + ["false_hit_rate"]
    lines = [
        "",
        "## 검증 프롬프트 버전 비교",
        "",
        f"- 즉시 hit 임계치(direct): {direct_threshold:.2f}. false_hit_rate는 첫 후보 하한"
        f"({cands[0]:.2f}) 기준(후보 하한이 낮을수록 검증 의존도가 커짐)",
        "",
        "| " + " | ".join(head) + " |",
        "| " + " | ".join("---" for _ in head) + " |",
    ]
    for version, recs in groups.items():
        stats = verify_stats(recs)
        rows = verify_sweep(recs, cands, direct_threshold=direct_threshold)
        label = version or "(미기록)"
        if not stats.get("total"):
            lines.append("| " + " | ".join([label] + ["-"] * (len(head) - 1)) + " |")
            continue
        cells = [
            label,
            _pct(stats["accuracy"]),
            str(len(stats["false_yes_ids"])),
            str(len(stats["false_no_ids"])),
        ]
        cells += [_pct(r["hit_rate"]) for r in rows]
        cells.append(_pct(rows[0]["false_hit_rate"]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines).rstrip() + "\n"

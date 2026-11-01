"""
RAG 검색 품질/접근 제어 평가용 순수 지표 함수 (X2).

API/DB/파일 I/O 없이 레코드만 다룹니다. `scripts/eval/rag_eval.py --live`가 수집한 jsonl을
`--from-cache`로 읽어 이 모듈이 집계/렌더링합니다. tests/test_rag_eval_metrics.py가 합성
레코드로 단위 테스트합니다.

레코드 형태:
{id, category, expected_doc_keys, forbidden_doc_keys,
 retrieved: [{doc_key, similarity, rank, substring_hit}], top_similarity,
 expected_substrings, margin}
- expected_substrings / substring_hit / margin 은 2회차(E3a)에서 추가된 선택 필드이며, 없는 레코드
  (1회차 jsonl)는 청크 적중·마진 집계에서 제외됩니다.
- --rerank 수집 레코드는 reranked(상위 top_k건, retrieved와 같은 dict 구조)·rerank_stage·
  rerank_latency_ms를 추가로 가지며, retrieved는 리랭크 전 후보 K건 전체(rank 1..K)입니다.
  compute_metrics(source="reranked")가 reranked로 집계하고, 키가 없는 레코드는 retrieved로 대체합니다.
- rank는 1부터 시작하며, 같은 doc_key의 여러 청크가 있으면 가장 높은 순위만 의미를 가집니다.
- expected_doc_keys가 비어 있는 케이스(acl 거절, out_of_corpus)는 recall/MRR/hit@1 집계에서 제외됩니다.
"""

OUT_OF_CORPUS = "out_of_corpus"
# ACL leak 집계 대상. distractor의 forbidden은 접근 가능한 문서라 노출돼도 leak이 아니며
# forbidden_exposed로 따로 셉니다.
ACL_CATEGORIES = ("acl_task", "acl_restricted")


def _ranked_keys(record: dict) -> list[str]:
    """rank 오름차순으로 중복 제거한 doc_key 목록."""
    seen: list[str] = []
    for item in sorted(record.get("retrieved", []), key=lambda r: r["rank"]):
        if item["doc_key"] not in seen:
            seen.append(item["doc_key"])
    return seen


def recall_at_k(record: dict, k: int) -> float | None:
    """상위 k개 문서 중 expected 문서 비율. expected가 비어 있으면 None."""
    expected = record["expected_doc_keys"]
    if not expected:
        return None
    top = set(_ranked_keys(record)[:k])
    return sum(1 for key in expected if key in top) / len(expected)


def hit_at_1(record: dict) -> bool | None:
    """1위 문서가 expected에 속하면 True. expected가 비어 있으면 None."""
    expected = record["expected_doc_keys"]
    if not expected:
        return None
    ranked = _ranked_keys(record)
    return bool(ranked) and ranked[0] in expected


def mrr(record: dict) -> float | None:
    """expected 중 가장 먼저 나온 문서의 역순위(1/rank). 못 찾으면 0.0, expected가 비면 None."""
    expected = record["expected_doc_keys"]
    if not expected:
        return None
    for pos, key in enumerate(_ranked_keys(record), start=1):
        if key in expected:
            return 1.0 / pos
    return 0.0


def chunk_hit_at_k(record: dict, k: int) -> float | None:
    """상위 k개 청크 중 정답 문구를 포함한 청크가 하나라도 있으면 1.0, 없으면 0.0.

    expected_substrings가 비어 있거나 없는 레코드는 채점 대상이 아니므로 None.
    """
    if not record.get("expected_substrings"):
        return None
    top = [item for item in record.get("retrieved", []) if item["rank"] <= k]
    return 1.0 if any(item.get("substring_hit") for item in top) else 0.0


def margin(record: dict) -> float | None:
    """1위와 2위 청크의 유사도 차이. 결과가 2건 미만이면 None."""
    ordered = sorted(record.get("retrieved", []), key=lambda r: r["rank"])
    if len(ordered) < 2:
        return None
    return ordered[0]["similarity"] - ordered[1]["similarity"]


def _forbidden_second(record: dict) -> bool:
    """1위 청크가 정답 문서이고 2위 청크가 forbidden 문서이면 True."""
    ordered = sorted(record.get("retrieved", []), key=lambda r: r["rank"])
    if len(ordered) < 2:
        return False
    return (
        ordered[0]["doc_key"] in record["expected_doc_keys"]
        and ordered[1]["doc_key"] in record.get("forbidden_doc_keys", [])
    )


def forbidden_exposed(record: dict) -> bool:
    """forbidden 문서가 검색 결과에 하나라도 나타나면 True(카테고리 무관)."""
    forbidden = set(record.get("forbidden_doc_keys", []))
    return any(item["doc_key"] in forbidden for item in record.get("retrieved", []))


def acl_leak(record: dict) -> bool:
    """ACL 카테고리 케이스에서 접근 불가 문서가 검색 결과에 나타나면 True."""
    return record.get("category") in ACL_CATEGORIES and forbidden_exposed(record)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _min(values: list[float]) -> float | None:
    return min(values) if values else None


def _aggregate(records: list[dict], k: int) -> dict:
    recalls = [v for r in records if (v := recall_at_k(r, k)) is not None]
    mrrs = [v for r in records if (v := mrr(r)) is not None]
    hits = [v for r in records if (v := hit_at_1(r)) is not None]
    chunk_hits = [v for r in records if (v := chunk_hit_at_k(r, k)) is not None]
    margins = [v for r in records if (v := margin(r)) is not None]
    return {
        "n": len(records),
        "n_scored": len(recalls),
        "recall_at_k": _mean(recalls),
        "mrr": _mean(mrrs),
        "hit_at_1": _mean([1.0 if h else 0.0 for h in hits]),
        "n_chunk_scored": len(chunk_hits),
        "chunk_hit_at_k": _mean(chunk_hits),
        "margin_mean": _mean(margins),
        "margin_min": _min(margins),
        "leak_count": sum(1 for r in records if acl_leak(r)),
        "forbidden_exposed_count": sum(1 for r in records if forbidden_exposed(r)),
    }


def _with_source(records: list[dict], source: str) -> list[dict]:
    """source 키의 검색 결과를 retrieved 자리에 넣은 레코드 사본을 만듭니다(없으면 retrieved 유지)."""
    if source == "retrieved":
        return records
    derived = []
    for record in records:
        items = record.get(source)
        if items is None:
            derived.append(record)
            continue
        derived.append(
            {
                **record,
                "retrieved": items,
                "top_similarity": max((r["similarity"] for r in items), default=0.0),
            }
        )
    return derived


def compute_metrics(records: list[dict], k: int = 5, source: str = "retrieved") -> dict:
    """전체/카테고리별 recall@k, MRR, hit@1, ACL leak 건수, forbidden 노출 건수를 계산합니다.

    out_of_corpus 카테고리에는 top_similarity의 mean/max를, distractor 카테고리에는 정답이 1위인
    케이스의 마진 평균(distractor_margin_mean)과 정답 1위·forbidden 2위 케이스의 마진
    (forbidden_second_count / forbidden_second_margin_mean)을 추가합니다.
    청크 적중(chunk_hit_at_k)은 expected_substrings가 있는 레코드만, 마진은 결과 2건 이상인 레코드만 채점합니다.
    source는 집계할 검색 결과 키("retrieved" 또는 "reranked")이며, 레코드에 그 키가 없으면 "retrieved"를 씁니다.
    """
    records = _with_source(records, source)
    by_category: dict[str, list[dict]] = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record)

    categories = {}
    for name, items in by_category.items():
        agg = _aggregate(items, k)
        if name == OUT_OF_CORPUS:
            sims = [r.get("top_similarity", 0.0) for r in items]
            agg["top_similarity_mean"] = _mean(sims)
            agg["top_similarity_max"] = max(sims) if sims else None
        if name == "distractor":
            top1 = [v for r in items if hit_at_1(r) and (v := margin(r)) is not None]
            second = [margin(r) for r in items if _forbidden_second(r) and margin(r) is not None]
            agg["distractor_margin_mean"] = _mean(top1)
            agg["distractor_margin_min"] = _min(top1)
            agg["forbidden_second_count"] = len(second)
            agg["forbidden_second_margin_mean"] = _mean(second)
        categories[name] = agg

    return {"k": k, "overall": _aggregate(records, k), "categories": categories}


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def _fmt_chunk_hit(agg: dict) -> str:
    """청크 hit@k를 '비율 (채점 n)'으로 표기합니다. 채점 대상이 없으면 '-'."""
    if agg["chunk_hit_at_k"] is None:
        return "-"
    return f"{_fmt(agg['chunk_hit_at_k'])} ({agg['n_chunk_scored']})"


def _fmt_margin(agg: dict) -> str:
    if agg["margin_mean"] is None:
        return "-"
    return f"{agg['margin_mean']:.3f}/{agg['margin_min']:.3f}"


def render_markdown(metrics: dict) -> str:
    """compute_metrics 결과를 markdown 표로 렌더링합니다."""
    k = metrics["k"]
    lines = [
        f"| 카테고리 | n | 채점 n | recall@{k} | MRR | hit@1 | ACL leak | forbidden 노출 "
        f"| 청크 hit@{k} | 마진 평균/최소 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    def row(name: str, agg: dict) -> str:
        return (
            f"| {name} | {agg['n']} | {agg['n_scored']} | {_fmt(agg['recall_at_k'])} "
            f"| {_fmt(agg['mrr'], pct=False)} | {_fmt(agg['hit_at_1'])} | {agg['leak_count']} "
            f"| {agg['forbidden_exposed_count']} | {_fmt_chunk_hit(agg)} | {_fmt_margin(agg)} |"
        )

    lines.append(row("overall", metrics["overall"]))
    for name, agg in sorted(metrics["categories"].items()):
        lines.append(row(name, agg))

    dis = metrics["categories"].get("distractor")
    if dis:
        lines += [
            "",
            f"distractor margin(정답 1위): mean={_fmt(dis['distractor_margin_mean'], pct=False)}, "
            f"min={_fmt(dis['distractor_margin_min'], pct=False)}; "
            f"forbidden 2위 {dis['forbidden_second_count']}건 "
            f"mean={_fmt(dis['forbidden_second_margin_mean'], pct=False)}",
        ]

    ooc = metrics["categories"].get(OUT_OF_CORPUS)
    if ooc:
        lines += [
            "",
            f"out_of_corpus top similarity: mean={_fmt(ooc['top_similarity_mean'], pct=False)}, "
            f"max={_fmt(ooc['top_similarity_max'], pct=False)}",
        ]
    return "\n".join(lines)


def has_reranked(records: list[dict]) -> bool:
    """레코드 중 하나라도 reranked 결과를 가지면 True."""
    return any("reranked" in r for r in records)


def render_diff_markdown(baseline: dict, reranked: dict) -> str:
    """기준선 대비 리랭크의 카테고리별 recall@k/MRR/hit@1 차이(리랭크 - 기준선) 표를 렌더링합니다."""
    k = reranked["k"]

    def delta(a: float | None, b: float | None, pct: bool) -> str:
        if a is None or b is None:
            return "-"
        d = b - a
        return f"{d * 100:+.1f}%p" if pct else f"{d:+.3f}"

    lines = [
        f"| 카테고리 | recall@{k} 차이 | MRR 차이 | hit@1 차이 |",
        "| --- | --- | --- | --- |",
    ]

    def row(name: str, base: dict, rer: dict) -> str:
        return (
            f"| {name} | {delta(base['recall_at_k'], rer['recall_at_k'], True)} "
            f"| {delta(base['mrr'], rer['mrr'], False)} "
            f"| {delta(base['hit_at_1'], rer['hit_at_1'], True)} |"
        )

    lines.append(row("overall", baseline["overall"], reranked["overall"]))
    for name in sorted(reranked["categories"]):
        if name in baseline["categories"]:
            lines.append(row(name, baseline["categories"][name], reranked["categories"][name]))
    return "\n".join(lines)

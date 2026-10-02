"""
RAG 검색 품질/접근 제어 평가용 순수 지표 함수 (X2).

API/DB/파일 I/O 없이 레코드만 다룹니다. `scripts/eval/rag_eval.py --live`가 수집한 jsonl을
`--from-cache`로 읽어 이 모듈이 집계/렌더링합니다. tests/test_rag_eval_metrics.py가 합성
레코드로 단위 테스트합니다.

레코드 형태:
{id, category, expected_doc_keys, forbidden_doc_keys,
 retrieved: [{doc_key, similarity, rank}], top_similarity}
- rank는 1부터 시작하며, 같은 doc_key의 여러 청크가 있으면 가장 높은 순위만 의미를 가집니다.
- expected_doc_keys가 비어 있는 케이스(acl 거절, out_of_corpus)는 recall/MRR/hit@1 집계에서 제외됩니다.
"""

OUT_OF_CORPUS = "out_of_corpus"


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


def acl_leak(record: dict) -> bool:
    """forbidden 문서가 검색 결과에 하나라도 나타나면 True."""
    forbidden = set(record.get("forbidden_doc_keys", []))
    return any(item["doc_key"] in forbidden for item in record.get("retrieved", []))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _aggregate(records: list[dict], k: int) -> dict:
    recalls = [v for r in records if (v := recall_at_k(r, k)) is not None]
    mrrs = [v for r in records if (v := mrr(r)) is not None]
    hits = [v for r in records if (v := hit_at_1(r)) is not None]
    return {
        "n": len(records),
        "n_scored": len(recalls),
        "recall_at_k": _mean(recalls),
        "mrr": _mean(mrrs),
        "hit_at_1": _mean([1.0 if h else 0.0 for h in hits]),
        "leak_count": sum(1 for r in records if acl_leak(r)),
    }


def compute_metrics(records: list[dict], k: int = 5) -> dict:
    """전체/카테고리별 recall@k, MRR, hit@1, ACL leak 건수를 계산합니다.

    out_of_corpus 카테고리에는 top_similarity의 mean/max를 추가합니다.
    """
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
        categories[name] = agg

    return {"k": k, "overall": _aggregate(records, k), "categories": categories}


def _fmt(value: float | None, pct: bool = True) -> str:
    if value is None:
        return "-"
    return f"{value * 100:.1f}%" if pct else f"{value:.3f}"


def render_markdown(metrics: dict) -> str:
    """compute_metrics 결과를 markdown 표로 렌더링합니다."""
    k = metrics["k"]
    lines = [
        f"| 카테고리 | n | 채점 n | recall@{k} | MRR | hit@1 | ACL leak |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    def row(name: str, agg: dict) -> str:
        return (
            f"| {name} | {agg['n']} | {agg['n_scored']} | {_fmt(agg['recall_at_k'])} "
            f"| {_fmt(agg['mrr'], pct=False)} | {_fmt(agg['hit_at_1'])} | {agg['leak_count']} |"
        )

    lines.append(row("overall", metrics["overall"]))
    for name, agg in sorted(metrics["categories"].items()):
        lines.append(row(name, agg))

    ooc = metrics["categories"].get(OUT_OF_CORPUS)
    if ooc:
        lines += [
            "",
            f"out_of_corpus top similarity: mean={_fmt(ooc['top_similarity_mean'], pct=False)}, "
            f"max={_fmt(ooc['top_similarity_max'], pct=False)}",
        ]
    return "\n".join(lines)

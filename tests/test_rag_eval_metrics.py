"""
scripts/eval/rag_eval_metrics.py 및 rag_eval.py CLI 단위 테스트 (합성 레코드, API/DB 호출 없음).
"""

import json

import pytest

from scripts.eval.rag_eval_metrics import (
    acl_leak,
    compute_metrics,
    hit_at_1,
    mrr,
    recall_at_k,
    render_markdown,
)


def _rec(id, category, expected, retrieved_keys, forbidden=(), top=None):
    retrieved = [
        {"doc_key": k, "similarity": 0.9 - 0.1 * i, "rank": i + 1} for i, k in enumerate(retrieved_keys)
    ]
    return {
        "id": id,
        "category": category,
        "expected_doc_keys": expected,
        "forbidden_doc_keys": list(forbidden),
        "retrieved": retrieved,
        "top_similarity": top if top is not None else (retrieved[0]["similarity"] if retrieved else 0.0),
    }


def test_recall_hit_mrr_single():
    r = _rec("a", "single_doc", ["d1"], ["d2", "d1", "d3"])
    assert recall_at_k(r, 1) == 0.0
    assert recall_at_k(r, 2) == 1.0
    assert hit_at_1(r) is False
    assert mrr(r) == pytest.approx(0.5)


def test_recall_partial_cross_doc():
    r = _rec("b", "cross_doc", ["d1", "d2"], ["d1", "d3", "d4"])
    assert recall_at_k(r, 3) == 0.5
    assert hit_at_1(r) is True
    assert mrr(r) == 1.0


def test_not_found_and_empty_expected():
    r = _rec("c", "single_doc", ["d1"], ["d2"])
    assert mrr(r) == 0.0 and recall_at_k(r, 5) == 0.0
    e = _rec("d", "acl_task", [], ["d2"], forbidden=["d9"])
    assert recall_at_k(e, 5) is None and mrr(e) is None and hit_at_1(e) is None


def test_duplicate_chunks_collapse_to_doc_rank():
    r = _rec("e", "single_doc", ["d2"], ["d1", "d1", "d2"])
    assert mrr(r) == pytest.approx(0.5)


def test_acl_leak():
    assert acl_leak(_rec("f", "acl_task", [], ["d1", "d9"], forbidden=["d9"])) is True
    assert acl_leak(_rec("g", "acl_task", [], ["d1"], forbidden=["d9"])) is False
    assert acl_leak(_rec("h", "single_doc", ["d1"], ["d1"])) is False


def test_compute_metrics_aggregation():
    records = [
        _rec("a", "single_doc", ["d1"], ["d1"]),
        _rec("b", "single_doc", ["d1"], ["d2", "d1"]),
        _rec("c", "acl_task", [], ["d9"], forbidden=["d9"]),
        _rec("d", "acl_task", [], [], forbidden=["d9"]),
        _rec("e", "out_of_corpus", [], ["d3"], top=0.4),
        _rec("f", "out_of_corpus", [], ["d3"], top=0.2),
    ]
    m = compute_metrics(records, k=5)
    single = m["categories"]["single_doc"]
    assert single["n"] == 2 and single["n_scored"] == 2
    assert single["recall_at_k"] == 1.0
    assert single["hit_at_1"] == 0.5
    assert single["mrr"] == pytest.approx(0.75)
    acl = m["categories"]["acl_task"]
    assert acl["leak_count"] == 1 and acl["recall_at_k"] is None
    ooc = m["categories"]["out_of_corpus"]
    assert ooc["top_similarity_mean"] == pytest.approx(0.3)
    assert ooc["top_similarity_max"] == 0.4
    assert m["overall"]["leak_count"] == 1 and m["overall"]["n"] == 6


def test_render_markdown():
    records = [
        _rec("a", "single_doc", ["d1"], ["d1"]),
        _rec("e", "out_of_corpus", [], ["d3"], top=0.4),
    ]
    md = render_markdown(compute_metrics(records, k=5))
    assert "recall@5" in md
    assert "| overall |" in md
    assert "| single_doc | 1 | 1 | 100.0% | 1.000 | 100.0% | 0 |" in md
    assert "| out_of_corpus |" in md
    assert "mean=0.400" in md


def test_rag_eval_cli_requires_mode_and_from_cache(tmp_path, capsys):
    from scripts.eval.rag_eval import main

    with pytest.raises(SystemExit):
        main([])

    cache = tmp_path / "rag.jsonl"
    cache.write_text(json.dumps(_rec("a", "single_doc", ["d1"], ["d1"]), ensure_ascii=False) + "\n", encoding="utf-8")
    assert main(["--from-cache", str(cache)]) == 0
    assert "| single_doc |" in capsys.readouterr().out


def test_rag_eval_live_blocked_without_gate(monkeypatch, capsys):
    from scripts.eval.rag_eval import main

    monkeypatch.delenv("RUN_LIVE_LLM", raising=False)
    assert main(["--live"]) == 1

"""
scripts/eval/rag_eval_metrics.py 및 rag_eval.py CLI 단위 테스트 (합성 레코드, API/DB 호출 없음).
"""

import json

import pytest

from scripts.eval.rag_eval_metrics import (
    acl_leak,
    chunk_hit_at_k,
    compute_metrics,
    forbidden_exposed,
    hit_at_1,
    margin,
    mrr,
    recall_at_k,
    render_markdown,
)


def _rec(id, category, expected, retrieved_keys, forbidden=(), top=None, subs=None, hits=(), sims=None):
    sims = sims if sims is not None else [0.9 - 0.1 * i for i in range(len(retrieved_keys))]
    retrieved = [
        {"doc_key": k, "similarity": sims[i], "rank": i + 1, "substring_hit": (i + 1) in hits}
        for i, k in enumerate(retrieved_keys)
    ]
    return {
        "id": id,
        "category": category,
        "expected_doc_keys": expected,
        "forbidden_doc_keys": list(forbidden),
        "expected_substrings": list(subs or []),
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
    assert acl_leak(_rec("i", "acl_restricted", [], ["d9"], forbidden=["d9"])) is True


def test_distractor_forbidden_is_exposure_not_leak():
    r = _rec("j", "distractor", ["d1"], ["d1", "d2"], forbidden=["d2"])
    assert acl_leak(r) is False
    assert forbidden_exposed(r) is True
    m = compute_metrics([r, _rec("k", "acl_task", [], ["d9"], forbidden=["d9"])], k=5)
    assert m["categories"]["distractor"]["leak_count"] == 0
    assert m["categories"]["distractor"]["forbidden_exposed_count"] == 1
    assert m["overall"]["leak_count"] == 1 and m["overall"]["forbidden_exposed_count"] == 2


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


def test_chunk_hit_at_k():
    # expected_substrings가 없으면 채점 제외(None)
    assert chunk_hit_at_k(_rec("a", "single_doc", ["d1"], ["d1"]), 5) is None
    # 레거시 레코드(필드 자체가 없음)도 None
    legacy = _rec("b", "single_doc", ["d1"], ["d1"])
    del legacy["expected_substrings"]
    assert chunk_hit_at_k(legacy, 5) is None
    r = _rec("c", "single_doc", ["d1"], ["d1", "d1", "d2"], subs=["x"], hits=(3,))
    assert chunk_hit_at_k(r, 2) == 0.0
    assert chunk_hit_at_k(r, 3) == 1.0
    miss = _rec("d", "single_doc", ["d1"], ["d1"], subs=["x"])
    assert chunk_hit_at_k(miss, 5) == 0.0


def test_margin():
    assert margin(_rec("a", "single_doc", ["d1"], ["d1"])) is None
    assert margin(_rec("b", "single_doc", ["d1"], [])) is None
    r = _rec("c", "single_doc", ["d1"], ["d1", "d2", "d3"], sims=[0.80, 0.74, 0.5])
    assert margin(r) == pytest.approx(0.06)


def test_chunk_hit_and_margin_aggregation():
    records = [
        _rec("a", "single_doc", ["d1"], ["d1", "d2"], subs=["x"], hits=(1,), sims=[0.8, 0.7]),
        _rec("b", "single_doc", ["d1"], ["d2", "d1"], subs=["x"], sims=[0.6, 0.55]),
        _rec("c", "single_doc", ["d1"], ["d1"]),
        _rec("d", "distractor", ["d1"], ["d1", "d2"], forbidden=["d2"], sims=[0.80, 0.76]),
        _rec("e", "distractor", ["d1"], ["d2", "d1"], forbidden=["d2"], sims=[0.80, 0.78]),
        _rec("f", "distractor", ["d1"], ["d1", "d3"], forbidden=["d2"], sims=[0.80, 0.70]),
    ]
    m = compute_metrics(records, k=5)
    single = m["categories"]["single_doc"]
    assert single["n_chunk_scored"] == 2
    assert single["chunk_hit_at_k"] == pytest.approx(0.5)
    assert single["margin_mean"] == pytest.approx((0.1 + 0.05) / 2)  # 'c'는 결과 1건이라 제외
    assert single["margin_min"] == pytest.approx(0.05)
    dis = m["categories"]["distractor"]
    assert dis["chunk_hit_at_k"] is None
    # 정답이 1위인 케이스: d(0.04), f(0.10)
    assert dis["distractor_margin_mean"] == pytest.approx(0.07)
    assert dis["distractor_margin_min"] == pytest.approx(0.04)
    # 정답 1위 + forbidden 2위: d 한 건
    assert dis["forbidden_second_count"] == 1
    assert dis["forbidden_second_margin_mean"] == pytest.approx(0.04)
    assert m["overall"]["margin_min"] == pytest.approx(0.02)


def test_render_markdown_new_columns():
    records = [
        _rec("a", "single_doc", ["d1"], ["d1", "d2"], subs=["x"], hits=(1,), sims=[0.8, 0.7]),
        _rec("d", "distractor", ["d1"], ["d1", "d2"], forbidden=["d2"], sims=[0.80, 0.76]),
    ]
    md = render_markdown(compute_metrics(records, k=5))
    assert "청크 hit@5" in md and "마진 평균/최소" in md
    assert "| single_doc | 1 | 1 | 100.0% | 1.000 | 100.0% | 0 | 0 | 100.0% (1) | 0.100/0.100 |" in md
    assert "| distractor | 1 | 1 | 100.0% | 1.000 | 100.0% | 0 | 1 | - | 0.040/0.040 |" in md
    assert "distractor margin(정답 1위): mean=0.040, min=0.040; forbidden 2위 1건 mean=0.040" in md


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


def _reranked_rec(id, category, expected, retrieved_keys, reranked_keys):
    rec = _rec(id, category, expected, retrieved_keys)
    rec["reranked"] = [
        {"doc_key": k, "similarity": 0.9 - 0.1 * i, "rank": i + 1, "substring_hit": False}
        for i, k in enumerate(reranked_keys)
    ]
    return rec


def test_compute_metrics_source_reranked_and_fallback():
    with_rerank = _reranked_rec("a", "single_doc", ["d1"], ["x", "d1", "y"], ["d1", "x", "y"])
    without = _rec("b", "single_doc", ["d1"], ["x", "d1"])

    base = compute_metrics([with_rerank, without], k=5)
    rer = compute_metrics([with_rerank, without], k=5, source="reranked")

    assert base["overall"]["hit_at_1"] == 0.0
    # reranked가 있는 레코드는 그 순서로, 없는 레코드는 retrieved로 대체
    assert rer["overall"]["hit_at_1"] == 0.5
    assert rer["overall"]["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    # source 기본값은 기존 동작과 동일
    assert compute_metrics([with_rerank], k=5) == compute_metrics([with_rerank], k=5, source="retrieved")


def test_rag_eval_from_cache_prints_baseline_rerank_and_diff(tmp_path, capsys):
    from scripts.eval.rag_eval import main

    cache = tmp_path / "rag.jsonl"
    rec = _reranked_rec("a", "single_doc", ["d1"], ["x", "d1", "y"], ["d1", "x", "y"])
    cache.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")

    assert main(["--from-cache", str(cache)]) == 0
    out = capsys.readouterr().out
    assert "기준선 (retrieved)" in out and "리랭크 (reranked)" in out
    assert "차이 (리랭크 - 기준선)" in out
    assert "| single_doc | +0.0%p | +0.500 | +100.0%p |" in out


def test_rag_eval_rerank_requires_enabled_setting(monkeypatch, capsys):
    from app.config import get_settings
    from scripts.eval.rag_eval import main

    monkeypatch.delenv("RERANK_ENABLED", raising=False)
    get_settings.cache_clear()
    try:
        assert main(["--live", "--rerank"]) == 1
    finally:
        get_settings.cache_clear()
    assert "RERANK_ENABLED" in capsys.readouterr().err

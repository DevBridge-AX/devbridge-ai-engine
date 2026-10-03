"""
scripts/eval/grounding_eval.py::compute_metrics 단위 테스트.

합성 레코드(API/DB 호출 없음)로 집계 로직만 검증합니다. 레코드 스키마는
scripts/eval/grounding_eval.py 모듈 docstring 및 --live 수집 로직과 동일합니다.
"""

import pytest

from scripts.eval.grounding_eval import (
    VARIANTS,
    compute_metrics,
    compute_repeat_stats,
    compute_variant_comparison,
    group_by_repeat,
    group_by_variant,
    render_repeat_markdown,
    render_report,
    render_sweep_markdown,
    render_variant_markdown,
)


def _record(
    *,
    id="c1",
    category="project_answerable",
    expected_groundable=True,
    top_similarity=0.5,
    llm_is_groundable=True,
    llm_confidence=0.9,
    parse_ok=True,
    fallback_reason=None,
    latency_ms=100.0,
    prompt_tokens=50,
    completion_tokens=10,
):
    return {
        "id": id,
        "category": category,
        "expected_groundable": expected_groundable,
        "top_similarity": top_similarity,
        "llm_is_groundable": llm_is_groundable,
        "llm_confidence": llm_confidence,
        "parse_ok": parse_ok,
        "fallback_reason": fallback_reason,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def test_empty_records_returns_all_none():
    m = compute_metrics([], 0.35)
    assert m["total"] == 0
    assert m["accuracy"] is None
    assert m["not_groundable"] == {"precision": None, "recall": None, "f1": None}
    assert m["filter_false_block_rate"] == {"overall": None, "by_category": {}}
    assert m["llm_call_saving_rate"] is None
    assert m["llm_only_agreement"] is None
    assert m["parse_failure_rate"] is None
    assert set(m["confidence_bucket_accuracy"]) == {"0.0-0.5", "0.5-0.8", "0.8-1.0"}
    assert all(v is None for v in m["confidence_bucket_accuracy"].values())


def test_accuracy_all_correct():
    records = [
        _record(id="a", expected_groundable=True, top_similarity=0.5, llm_is_groundable=True),
        _record(id="b", expected_groundable=False, top_similarity=0.1, llm_is_groundable=False),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["total"] == 2
    assert m["accuracy"] == 1.0


def test_accuracy_partial_mismatch():
    records = [
        # 1차 필터 통과, LLM=True, 라벨 True -> 정답
        _record(id="a", expected_groundable=True, top_similarity=0.5, llm_is_groundable=True),
        # 1차 필터 통과, LLM=True, 라벨 False -> 오답(LLM이 groundable로 오판)
        _record(id="b", expected_groundable=False, top_similarity=0.6, llm_is_groundable=True),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["accuracy"] == 0.5


def test_not_groundable_precision_recall_f1():
    records = [
        # 예측 not-groundable(final=False), 실제 not-groundable -> TP
        _record(id="tp", expected_groundable=False, top_similarity=0.1),
        # 예측 not-groundable(1차 필터 차단), 실제 groundable -> FP
        _record(id="fp", expected_groundable=True, top_similarity=0.1),
        # 예측 groundable, 실제 not-groundable -> FN
        _record(
            id="fn",
            expected_groundable=False,
            top_similarity=0.6,
            llm_is_groundable=True,
        ),
        # 예측 groundable, 실제 groundable -> TN(precision/recall 분모에 영향 없음)
        _record(id="tn", expected_groundable=True, top_similarity=0.6, llm_is_groundable=True),
    ]
    m = compute_metrics(records, threshold=0.35)
    ng = m["not_groundable"]
    # TP=1, FP=1, FN=1
    assert ng["precision"] == 0.5
    assert ng["recall"] == 0.5
    assert ng["f1"] == 0.5


def test_filter_false_block_rate_by_category():
    records = [
        # chitchat, 라벨 groundable인데 유사도 미달 -> false block
        _record(id="c1", category="chitchat", expected_groundable=True, top_similarity=0.05),
        _record(id="c2", category="chitchat", expected_groundable=True, top_similarity=0.9, llm_is_groundable=True),
        # project_answerable, 라벨 groundable이고 유사도 통과 -> false block 아님
        _record(
            id="p1",
            category="project_answerable",
            expected_groundable=True,
            top_similarity=0.9,
            llm_is_groundable=True,
        ),
    ]
    m = compute_metrics(records, threshold=0.35)
    by_cat = m["filter_false_block_rate"]["by_category"]
    assert by_cat["chitchat"] == 0.5  # 2건 중 1건 false block
    assert by_cat["project_answerable"] == 0.0
    assert m["filter_false_block_rate"]["overall"] == 1 / 3


def test_llm_call_saving_rate():
    records = [
        _record(id="a", top_similarity=0.1),  # below threshold
        _record(id="b", top_similarity=0.5),  # above threshold
        _record(id="c", top_similarity=0.9),  # above threshold
        _record(id="d", top_similarity=0.2),  # below threshold
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["llm_call_saving_rate"] == 0.5


def test_llm_only_agreement_ignores_threshold():
    """1차 필터로 차단되는 케이스(top_similarity<threshold)도 LLM 단독 일치율 계산에는 포함됩니다."""
    records = [
        # 유사도는 낮아 1차 필터에서 걸리지만, LLM 판정 자체는 라벨과 일치
        _record(id="a", expected_groundable=False, top_similarity=0.05, llm_is_groundable=False),
        _record(id="b", expected_groundable=True, top_similarity=0.05, llm_is_groundable=True),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["llm_only_agreement"] == 1.0
    # 하지만 최종 accuracy(1차 필터 포함)는 낮아야 함: 둘 다 top_similarity<threshold라
    # 최종 판정은 강제로 False가 되어 "a"만 라벨과 일치.
    assert m["accuracy"] == 0.5


def test_fail_open_fallback_counts_as_groundable():
    """fallback_reason이 있으면(llm_error/parse_error) 실제 llm_is_groundable 값과 무관하게
    fail-open으로 groundable=True 취급되어야 합니다 (app/core/rag/grounding.py::assess())."""
    records = [
        _record(
            id="a",
            expected_groundable=True,
            top_similarity=0.9,
            llm_is_groundable=None,
            llm_confidence=None,
            parse_ok=False,
            fallback_reason="llm_error",
        ),
        _record(
            id="b",
            expected_groundable=True,
            top_similarity=0.9,
            llm_is_groundable=None,
            llm_confidence=None,
            parse_ok=False,
            fallback_reason="parse_error",
        ),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["accuracy"] == 1.0
    assert m["llm_only_agreement"] == 1.0
    assert m["parse_failure_rate"] == 1.0


def test_parse_failure_rate():
    records = [
        _record(id="a", parse_ok=True),
        _record(id="b", parse_ok=False, fallback_reason="parse_error", llm_is_groundable=None, llm_confidence=None),
        _record(id="c", parse_ok=True),
        _record(id="d", parse_ok=False, fallback_reason="llm_error", llm_is_groundable=None, llm_confidence=None),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["parse_failure_rate"] == 0.5


def test_confidence_bucket_accuracy():
    records = [
        _record(id="a", expected_groundable=True, top_similarity=0.9, llm_is_groundable=True, llm_confidence=0.2),
        _record(id="b", expected_groundable=False, top_similarity=0.9, llm_is_groundable=True, llm_confidence=0.3),
        _record(id="c", expected_groundable=True, top_similarity=0.9, llm_is_groundable=True, llm_confidence=0.6),
        _record(id="d", expected_groundable=True, top_similarity=0.9, llm_is_groundable=True, llm_confidence=0.95),
        _record(id="e", expected_groundable=True, top_similarity=0.9, llm_is_groundable=True, llm_confidence=1.0),
    ]
    m = compute_metrics(records, threshold=0.35)
    bucket = m["confidence_bucket_accuracy"]
    # 0.0-0.5: a(맞음), b(라벨 False인데 groundable=True로 오답) -> 1/2
    assert bucket["0.0-0.5"] == 0.5
    # 0.5-0.8: c만 -> 1/1
    assert bucket["0.5-0.8"] == 1.0
    # 0.8-1.0: d, e(경계값 1.0 포함) -> 2/2
    assert bucket["0.8-1.0"] == 1.0


def test_confidence_bucket_excludes_fallback_records():
    records = [
        _record(
            id="a",
            parse_ok=False,
            fallback_reason="parse_error",
            llm_is_groundable=None,
            llm_confidence=None,
        ),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert all(v is None for v in m["confidence_bucket_accuracy"].values())


def test_mean_tokens_and_latency():
    records = [
        _record(id="a", latency_ms=100.0, prompt_tokens=40, completion_tokens=10),
        _record(id="b", latency_ms=300.0, prompt_tokens=60, completion_tokens=20),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["mean_latency_ms"] == 200.0
    assert m["mean_prompt_tokens"] == 50.0
    assert m["mean_completion_tokens"] == 15.0


def test_mean_tokens_none_when_missing():
    records = [
        _record(id="a", latency_ms=None, prompt_tokens=None, completion_tokens=None),
    ]
    m = compute_metrics(records, threshold=0.35)
    assert m["mean_latency_ms"] is None
    assert m["mean_prompt_tokens"] is None
    assert m["mean_completion_tokens"] is None


def test_render_sweep_markdown_contains_thresholds_and_categories():
    records = [
        _record(id="a", category="chitchat", expected_groundable=True, top_similarity=0.05),
        _record(id="b", category="project_answerable", expected_groundable=True, top_similarity=0.9, llm_is_groundable=True),
    ]
    text = render_sweep_markdown(records, thresholds=[0.2, 0.35, 0.6])
    assert "0.20" in text
    assert "0.35" in text
    assert "0.60" in text
    assert "chitchat" in text
    assert "project_answerable" in text
    assert "케이스 수: 2" in text


def test_render_sweep_markdown_empty_records():
    text = render_sweep_markdown([], thresholds=[0.35])
    assert "케이스 수: 0" in text


# ---------------------------------------------------------------------------
# 판정 변형 비교 (G1)
# ---------------------------------------------------------------------------

def _variant_data():
    base = [
        _record(id="g1", expected_groundable=True, llm_is_groundable=True, prompt_tokens=100, latency_ms=100.0),
        _record(id="g2", expected_groundable=True, llm_is_groundable=True, prompt_tokens=100, latency_ms=100.0),
        _record(id="n1", category="project_unanswerable", expected_groundable=False,
                llm_is_groundable=True, prompt_tokens=100, latency_ms=100.0),
        _record(id="n2", category="project_unanswerable", expected_groundable=False,
                llm_is_groundable=False, prompt_tokens=100, latency_ms=100.0),
    ]
    strict = [
        _record(id="g1", expected_groundable=True, llm_is_groundable=True, prompt_tokens=60, latency_ms=50.0),
        _record(id="g2", expected_groundable=True, llm_is_groundable=False, prompt_tokens=60, latency_ms=50.0),
        _record(id="n1", category="project_unanswerable", expected_groundable=False,
                llm_is_groundable=False, prompt_tokens=60, latency_ms=50.0),
        _record(id="n2", category="project_unanswerable", expected_groundable=False,
                llm_is_groundable=None, prompt_tokens=60, latency_ms=50.0, parse_ok=False,
                llm_confidence=None, fallback_reason="parse_error"),
    ]
    return {"baseline": base, "strict": strict}


def test_group_by_variant_defaults_to_baseline():
    grouped = group_by_variant([_record(id="a"), {**_record(id="b"), "variant": "strict"}])
    assert set(grouped) == {"baseline", "strict"}
    assert [r["id"] for r in grouped["baseline"]] == ["a"]


def test_variants_table_matches_spec():
    assert set(VARIANTS) == {"baseline", "strict", "top3", "strict_top3", "strict_top3_cap"}
    assert VARIANTS["strict_top3_cap"] == {"prompt_version": "v2-strict", "top_k": 3, "max_chunk_chars": 600}


def test_compute_variant_comparison_metrics():
    result = compute_variant_comparison(_variant_data(), threshold=0.35)
    b = result["variants"]["baseline"]
    s = result["variants"]["strict"]

    assert b["accuracy"] == 0.75
    assert b["recall"] == 0.5  # n2만 not-groundable로 잡음
    assert b["precision"] == 1.0
    assert b["groundable_false_block_count"] == 0
    assert b["groundable_total"] == 2
    assert b["mean_prompt_tokens"] == 100.0
    assert b["parse_failure_rate"] == 0.0

    # strict: g2 오차단, n2는 parse_error fail-open -> groundable(오답)
    assert s["groundable_false_block_count"] == 1
    assert s["groundable_false_block_rate"] == 0.5
    assert s["recall"] == 0.5
    assert s["precision"] == 0.5
    assert s["mean_prompt_tokens"] == 60.0
    assert s["mean_latency_ms"] == 50.0
    assert s["parse_failure_rate"] == 0.25


def test_compute_variant_comparison_disagreements():
    result = compute_variant_comparison(_variant_data(), threshold=0.35)
    by_id = {d["id"]: d for d in result["disagreements"]}
    assert set(by_id) == {"g2", "n1", "n2"}
    assert by_id["n2"]["correct"] == {"baseline": True, "strict": False}  # strict는 fail-open
    assert by_id["g2"]["correct"] == {"baseline": True, "strict": False}
    assert by_id["n1"]["correct"] == {"baseline": False, "strict": True}
    assert by_id["n1"]["category"] == "project_unanswerable"
    assert by_id["n1"]["expected"] is False


def test_compute_variant_comparison_threshold_applies_filter():
    result = compute_variant_comparison(_variant_data(), threshold=0.95)
    # 모든 top_similarity(0.5)가 임계치 미만 -> 전부 1차 필터 차단, variant 간 차이 없음
    assert result["variants"]["baseline"]["groundable_false_block_count"] == 2
    assert result["disagreements"] == []


def test_render_variant_markdown_contents():
    text = render_variant_markdown(_variant_data(), threshold=0.35)
    assert "threshold 0.35" in text
    assert "| baseline |" in text and "| strict |" in text
    assert "1/2 (50.0%)" in text
    assert "| g2 | project_answerable | groundable | ✓ | ✗ |" in text
    assert "| n1 | project_unanswerable | not-groundable | ✗ | ✓ |" in text


def test_render_variant_markdown_no_disagreement():
    data = {"baseline": [_record(id="a")], "strict": [_record(id="a")]}
    assert "없음" in render_variant_markdown(data, threshold=0.35)


def test_render_report_single_variant_is_plain_sweep():
    text = render_report([_record(id="a")], threshold=0.35)
    assert "임계치 스윕" in text
    assert "변형 비교" not in text


def test_main_rejects_unknown_variant_via_argparse(capsys):
    from scripts.eval.grounding_eval import main

    with pytest.raises(SystemExit) as exc:
        main(["--live", "--variants", "baseline,nope"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "nope" in err
    assert "strict_top3" in err  # 유효한 키 안내


def test_render_report_multi_variant_includes_both():
    records = [{**r, "variant": name} for name, rs in _variant_data().items() for r in rs]
    text = render_report(records, threshold=0.35)
    assert "임계치 스윕" in text
    assert "변형 비교" in text


# ---------------------------------------------------------------------------
# 반복 측정 (L4, --repeat)
# ---------------------------------------------------------------------------

def _rep(rec: dict, repeat: int | None, variant: str = "baseline") -> dict:
    out = {**rec, "variant": variant}
    if repeat is not None:
        out["repeat"] = repeat
    return out


def _repeat_records():
    """케이스 4건 x 3회. g2만 repeat 1에서 판정이 뒤집힘(True->False)."""
    records = []
    for rep in range(3):
        records += [
            _rep(_record(id="g1", expected_groundable=True, llm_is_groundable=True), rep),
            _rep(_record(id="g2", expected_groundable=True, llm_is_groundable=(rep != 1)), rep),
            _rep(_record(id="n1", category="project_unanswerable", expected_groundable=False,
                         llm_is_groundable=False), rep),
            _rep(_record(id="n2", category="project_unanswerable", expected_groundable=False,
                         llm_is_groundable=False), rep),
        ]
    return records


def test_group_by_repeat_missing_field_is_zero():
    grouped = group_by_repeat([_record(id="a"), {**_record(id="b"), "repeat": 1}, {**_record(id="c"), "repeat": None}])
    assert sorted(grouped) == [0, 1]
    assert [r["id"] for r in grouped[0]] == ["a", "c"]


def test_compute_repeat_stats_mean_stdev_min_max():
    stats = compute_repeat_stats(_repeat_records(), 0.35)
    v = stats["variants"]["baseline"]
    assert v["repeats"] == 3 and v["n"] == 4
    acc = v["metrics"]["accuracy"]
    # repeat별 정확도: 1.0, 0.75, 1.0
    assert acc["mean"] == pytest.approx((1.0 + 0.75 + 1.0) / 3)
    assert acc["stdev"] == pytest.approx(statistics_stdev([1.0, 0.75, 1.0]))
    assert acc["min"] == 0.75 and acc["max"] == 1.0
    fb = v["metrics"]["groundable_false_block_rate"]
    assert fb["min"] == 0.0 and fb["max"] == 0.5  # repeat 1에서 g2 오차단 1/2
    # not-gr recall은 n1,n2 모두 항상 맞아 1.0
    assert v["metrics"]["recall"]["mean"] == 1.0 and v["metrics"]["recall"]["stdev"] == 0.0


def statistics_stdev(values):
    import statistics

    return statistics.stdev(values)


def test_compute_repeat_stats_single_repeat_stdev_none():
    records = [_rep(r, None) for r in _variant_data()["baseline"]]
    v = compute_repeat_stats(records, 0.35)["variants"]["baseline"]
    assert v["repeats"] == 1
    assert v["metrics"]["accuracy"]["stdev"] is None
    assert v["metrics"]["accuracy"]["mean"] == v["metrics"]["accuracy"]["min"] == v["metrics"]["accuracy"]["max"]
    assert v["unstable_case_ids"] == [] and v["flip_rate"] == 0.0


def test_compute_repeat_stats_flip_detection():
    v = compute_repeat_stats(_repeat_records(), 0.35)["variants"]["baseline"]
    assert v["unstable_case_ids"] == ["g2"]
    assert v["flip_rate"] == pytest.approx(0.25)


def test_compute_repeat_stats_flip_uses_threshold_and_present_repeats_only():
    # 낮은 유사도 케이스는 LLM 값이 흔들려도 1차 필터로 항상 False -> 안정
    records = [
        _rep(_record(id="low", top_similarity=0.1, llm_is_groundable=bool(rep)), rep) for rep in range(2)
    ]
    # 일부 repeat에만 존재하는 케이스는 있는 repeat끼리만 비교(단일 관측이면 불안정 아님)
    records.append(_rep(_record(id="only1", llm_is_groundable=True), 1))
    v = compute_repeat_stats(records, 0.35)["variants"]["baseline"]
    assert v["unstable_case_ids"] == []
    assert v["n"] == 2


def test_render_repeat_markdown_contents():
    text = render_repeat_markdown(_repeat_records(), 0.35)
    assert text.startswith("# 반복 측정 (3회, threshold 0.35)")
    assert "±" in text and "g2" in text and "25.0%" in text


def test_render_report_repeat_section_only_when_multi_repeat():
    multi = render_report(_repeat_records(), threshold=0.35)
    assert "# 반복 측정 (3회" in multi
    assert "repeat 0 레코드만" in multi
    assert "케이스 수: 4" in multi  # 스윕은 repeat 0만 -> 12가 아닌 4
    single = render_report([_rep(r, 0) for r in _variant_data()["baseline"]], threshold=0.35)
    assert "반복 측정" not in single


def test_render_report_single_repeat_identical_to_legacy_cache():
    legacy = [r for rs in _variant_data().values() for r in rs]  # 구 캐시: variant/repeat 없음
    legacy_multi = [{**r, "variant": name} for name, rs in _variant_data().items() for r in rs]
    with_repeat = [{**r, "repeat": 0} for r in legacy_multi]
    assert render_report(with_repeat, threshold=0.35) == render_report(legacy_multi, threshold=0.35)
    assert render_report([{**r, "repeat": 0} for r in legacy], threshold=0.35) == render_report(legacy, threshold=0.35)
    # 구 캐시 출력은 repeat 도입 전과 동일한 섹션 구성(반복 측정 없음)
    assert "반복 측정" not in render_report(legacy_multi, threshold=0.35)


def test_main_rejects_repeat_zero(capsys):
    from scripts.eval.grounding_eval import main

    with pytest.raises(SystemExit) as exc:
        main(["--live", "--repeat", "0"])
    assert exc.value.code == 2
    assert "--repeat" in capsys.readouterr().err


def test_main_rejects_repeat_with_from_cache(capsys, tmp_path):
    from scripts.eval.grounding_eval import main

    with pytest.raises(SystemExit) as exc:
        main(["--from-cache", str(tmp_path / "x.jsonl"), "--repeat", "2"])
    assert exc.value.code == 2
    assert "--live" in capsys.readouterr().err


def test_collect_records_repeat_retrieves_once_and_judges_n_times(monkeypatch, tmp_path):
    import asyncio
    import json
    from types import SimpleNamespace

    from app.config import get_settings
    from app.core.llm import provider as llm
    from app.core.rag import retriever
    from app.db.vector_store import get_vector_store
    import scripts.eval.seed as seed
    from scripts.eval.grounding_eval import _collect_records

    dataset = tmp_path / "cases.jsonl"
    cases = [
        {"id": "c1", "category": "project_answerable", "question": "q1", "expected_groundable": True},
        {"id": "c2", "category": "project_unanswerable", "question": "q2", "expected_groundable": False},
    ]
    dataset.write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases), encoding="utf-8")

    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "unused"))
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "unused-m"))

    calls = {"retrieve": 0, "judge": 0}

    class _Db:
        def close(self):
            pass

    async def fake_seed(_tmp):
        return _Db(), "ws-test"

    async def fake_retrieve(question, workspace_id, db, top_k=5):
        calls["retrieve"] += 1
        return [SimpleNamespace(similarity_score=0.6, content="c", title="t", source_type="document")]

    async def fake_judge(prompt, *, system_prompt=None):
        calls["judge"] += 1
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        return {"is_groundable": calls["judge"] % 2 == 0, "confidence": 0.9}, usage

    monkeypatch.setattr(seed, "seed_workspace_async", fake_seed)
    monkeypatch.setattr(retriever, "retrieve", fake_retrieve)
    monkeypatch.setattr(llm, "call_grounding", fake_judge)

    try:
        records = asyncio.run(_collect_records(dataset, None, ["baseline", "strict_top3"], repeat=2))
    finally:
        get_settings.cache_clear()
        get_vector_store.cache_clear()

    assert calls["retrieve"] == 2  # 케이스당 1회
    assert calls["judge"] == 2 * 2 * 2  # 케이스 x 변형 x repeat
    assert len(records) == 8
    assert sorted({r["repeat"] for r in records}) == [0, 1]
    assert {(r["variant"], r["id"], r["repeat"]) for r in records} == {
        (v, c, i) for v in ("baseline", "strict_top3") for c in ("c1", "c2") for i in (0, 1)
    }

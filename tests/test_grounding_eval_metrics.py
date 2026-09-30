"""
scripts/eval/grounding_eval.py::compute_metrics 단위 테스트.

합성 레코드(API/DB 호출 없음)로 집계 로직만 검증합니다. 레코드 스키마는
scripts/eval/grounding_eval.py 모듈 docstring 및 --live 수집 로직과 동일합니다.
"""

from scripts.eval.grounding_eval import (
    VARIANTS,
    compute_metrics,
    compute_variant_comparison,
    group_by_variant,
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


def test_render_report_multi_variant_includes_both():
    records = [{**r, "variant": name} for name, rs in _variant_data().items() for r in rs]
    text = render_report(records, threshold=0.35)
    assert "임계치 스윕" in text
    assert "변형 비교" in text

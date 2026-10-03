"""
scripts/eval/cache_eval_metrics.py 유닛 테스트 (A9).

전부 합성 데이터로 검증하며 실 API 호출이나 DB/벡터스토어 접근이 없습니다.
"""

import json
from pathlib import Path

import pytest

from scripts.eval.cache_eval_metrics import (
    cosine_similarity,
    load_cache,
    load_dataset,
    recommend_threshold,
    render_markdown,
    save_cache,
    similarity_distribution,
    sweep,
)

_DATASET = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "datasets" / "cache_pairs.jsonl"
_HOLDOUT = Path(__file__).resolve().parents[1] / "scripts" / "eval" / "datasets" / "cache_pairs_holdout.jsonl"


def _records() -> list[dict]:
    same = [0.99, 0.96, 0.90, 0.80]
    diff = [0.97, 0.92, 0.86, 0.70]
    records = []
    for i, s in enumerate(same):
        records.append({"id": f"s{i}", "kind": "same_intent", "q1": f"a{i}", "q2": f"b{i}", "similarity": s})
    for i, s in enumerate(diff):
        records.append({"id": f"d{i}", "kind": "different_intent", "q1": f"c{i}", "q2": f"e{i}", "similarity": s})
    return records


def _row(rows: list[dict], t: float) -> dict:
    return next(r for r in rows if r["threshold"] == t)


class TestCosine:
    def test_identical_vectors(self):
        assert cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            cosine_similarity([1.0], [1.0, 2.0])


class TestSweep:
    def test_default_thresholds_range(self):
        rows = sweep(_records())
        assert len(rows) == 15
        assert rows[0]["threshold"] == 0.85
        assert rows[-1]["threshold"] == 0.99

    def test_rates_at_thresholds(self):
        rows = sweep(_records(), [0.85, 0.90, 0.95, 0.98])

        r85 = _row(rows, 0.85)
        assert r85["hit_rate"] == pytest.approx(3 / 4)  # 0.99, 0.96, 0.90
        assert r85["false_hit_rate"] == pytest.approx(3 / 4)  # 0.97, 0.92, 0.86
        assert r85["precision"] == pytest.approx(3 / 6)
        assert (r85["same_hits"], r85["diff_hits"]) == (3, 3)

        r90 = _row(rows, 0.90)  # 경계값 포함(>=)
        assert r90["hit_rate"] == pytest.approx(3 / 4)
        assert r90["false_hit_rate"] == pytest.approx(2 / 4)

        r95 = _row(rows, 0.95)
        assert r95["hit_rate"] == pytest.approx(2 / 4)
        assert r95["false_hit_rate"] == pytest.approx(1 / 4)
        assert r95["precision"] == pytest.approx(2 / 3)
        assert r95["recall"] == r95["hit_rate"]
        assert r95["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / ((2 / 3) + 0.5))

        r98 = _row(rows, 0.98)
        assert r98["hit_rate"] == pytest.approx(1 / 4)
        assert r98["false_hit_rate"] == 0.0

    def test_no_hits_gives_none_precision_and_f1(self):
        rows = sweep(_records(), [1.0])
        assert rows[0]["hit_rate"] == 0.0
        assert rows[0]["precision"] is None
        assert rows[0]["f1"] is None


class TestRecommend:
    def test_strict_zero_false_hit(self):
        rows = sweep(_records(), [round(0.85 + 0.01 * i, 2) for i in range(15)])
        rec = recommend_threshold(rows, max_false_hit_rate=0.0)
        # false hit 0은 0.98 이상부터(0.97이 다른 의도 최대). 그중 hit_rate 최대는 0.98(1/4), 동률 시 낮은 값.
        assert rec is not None
        assert rec["threshold"] == 0.98
        assert rec["hit_rate"] == pytest.approx(1 / 4)

    def test_relaxed_limit_picks_higher_hit_rate(self):
        rows = sweep(_records())
        rec = recommend_threshold(rows, max_false_hit_rate=0.02)
        # 4건 중 1건(25%)이 허용 한도를 넘으므로 0%와 동일한 결과
        assert rec is not None and rec["threshold"] == 0.98
        rec_loose = recommend_threshold(rows, max_false_hit_rate=0.25)
        assert rec_loose is not None
        assert rec_loose["threshold"] == 0.93  # false 1/4 허용 -> hit 2/4, 동률 중 최저(0.92부터 false 2건)

    def test_none_when_unsatisfiable(self):
        rows = sweep(_records(), [0.85, 0.90])
        assert recommend_threshold(rows, max_false_hit_rate=0.0) is None


class TestDistribution:
    def test_per_kind_stats(self):
        dist = similarity_distribution(_records())
        same = dist["same_intent"]
        assert same["min"] == 0.80
        assert same["max"] == 0.99
        assert same["median"] == pytest.approx(0.93)
        assert same["mean"] == pytest.approx((0.99 + 0.96 + 0.90 + 0.80) / 4)
        assert dist["different_intent"]["max"] == 0.97
        assert set(same) == {"min", "p25", "median", "p75", "max", "mean"}


class TestRender:
    def test_markdown_contains_sections_and_recommendation(self):
        records = _records()
        rows = sweep(records)
        recs = {
            "false hit 0% 허용": recommend_threshold(rows, 0.0),
            "false hit 2% 이하 허용": recommend_threshold(rows, 0.02),
        }
        md = render_markdown(records, rows, recs, embedding_model="test-embed")
        for heading in ("## 실험 조건", "## 유사도 분포", "## 임계치 스윕", "## 권고 임계치", "## 경계 사례"):
            assert heading in md
        assert "test-embed" in md
        assert "**0.98**" in md
        assert "| threshold | hit_rate | false_hit_rate | precision | f1 | hits/same | hits/diff |" in md
        # 경계 사례: same 최저(s3 0.80)와 different 최고(d0 0.97)
        assert "s3" in md and "d0" in md

    def test_markdown_handles_no_recommendation(self):
        records = _records()
        md = render_markdown(records, sweep(records, [0.85]), {"엄격": None})
        assert "조건을 만족하는 임계치 없음" in md


class TestIO:
    def test_cache_roundtrip(self, tmp_path):
        path = tmp_path / "sub" / "cache.jsonl"
        save_cache(path, _records())
        assert load_cache(path) == _records()

    def test_load_cache_skips_bad_lines(self, tmp_path):
        path = tmp_path / "c.jsonl"
        path.write_text('{"id": "a"}\nnot json\n\n{"id": "b"}\n', encoding="utf-8")
        assert [r["id"] for r in load_cache(path)] == ["a", "b"]

    def test_load_dataset_rejects_invalid_kind(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(
            json.dumps({"id": "x", "kind": "maybe", "q1": "a", "q2": "b"}) + "\n", encoding="utf-8"
        )
        with pytest.raises(ValueError):
            load_dataset(path)

    def test_load_dataset_rejects_missing_field(self, tmp_path):
        path = tmp_path / "d.jsonl"
        path.write_text(json.dumps({"id": "x", "kind": "same_intent", "q1": "a"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_dataset(path)


class TestDatasetFile:
    def test_dataset_shape(self):
        pairs = load_dataset(_DATASET)
        assert len(pairs) == 40
        assert sum(1 for p in pairs if p["kind"] == "same_intent") == 20
        assert sum(1 for p in pairs if p["kind"] == "different_intent") == 20
        assert len({p["id"] for p in pairs}) == 40
        for p in pairs:
            assert p["q1"].strip() and p["q2"].strip()
            assert p["q1"] != p["q2"]


# ---------------------------------------------------------------------------
# LLM 재검증(verify) 지표
# ---------------------------------------------------------------------------

from scripts.eval.cache_eval_metrics import (  # noqa: E402
    has_verify_fields,
    render_verify_markdown,
    verify_stats,
    verify_sweep,
)


def _verify_records() -> list[dict]:
    # (id, kind, similarity, verify_same, latency, prompt_tokens)
    rows = [
        ("s0", "same_intent", 0.97, True, 500.0, 40),   # 즉시 hit
        ("s1", "same_intent", 0.90, True, 600.0, 40),   # 후보, YES
        ("s2", "same_intent", 0.87, False, 700.0, 40),  # 후보, 오판 NO
        ("s3", "same_intent", 0.82, True, 800.0, 40),   # 낮은 후보
        ("d0", "different_intent", 0.96, False, 400.0, 40),  # 즉시 hit = false hit (검증 무관)
        ("d1", "different_intent", 0.91, False, 450.0, 40),  # 후보, NO
        ("d2", "different_intent", 0.88, True, 550.0, 40),   # 후보, 오판 YES
        ("d3", "different_intent", 0.70, False, 300.0, 40),
    ]
    return [
        {
            "id": i, "kind": k, "q1": "a", "q2": "b", "similarity": sim,
            "verify_outcome": "yes" if v else "no", "verify_same": v,
            "verify_latency_ms": lat, "verify_prompt_tokens": pt, "verify_completion_tokens": 1,
            "verify_model": "haiku", "verify_prompt_version": "v1",
        }
        for i, k, sim, v, lat, pt in rows
    ]


def _vrow(rows, t):
    return next(r for r in rows if r["candidate_threshold"] == t)


def test_has_verify_fields_detects_old_cache():
    assert has_verify_fields(_verify_records()) is True
    assert has_verify_fields(_records()) is False


def test_verify_sweep_counts_and_rates():
    rows = verify_sweep(_verify_records(), [0.86, 0.80, 0.95], direct_threshold=0.95)

    r86 = _vrow(rows, 0.86)
    # same: s0 직접 hit, s1 YES hit, s2 NO miss, s3 구간 밖 -> 2/4
    assert (r86["same_hits"], r86["same_total"]) == (2, 4)
    # diff: d0 직접 hit(false hit), d1 NO, d2 YES(false hit), d3 miss -> 2/4
    assert (r86["diff_hits"], r86["diff_total"]) == (2, 4)
    assert r86["hit_rate"] == 0.5 and r86["false_hit_rate"] == 0.5
    assert (r86["verify_calls"], r86["verify_calls_same"], r86["verify_calls_diff"]) == (4, 2, 2)
    assert r86["adopt"] is False

    r80 = _vrow(rows, 0.80)
    assert r80["same_hits"] == 3  # s3도 후보 구간, YES
    assert r80["verify_calls"] == 5

    # candidate == direct 이면 후보 구간이 없어 검증 호출 0, 임계치만 쓴 것과 동일
    r95 = _vrow(rows, 0.95)
    assert r95["verify_calls"] == 0
    assert (r95["same_hits"], r95["diff_hits"]) == (1, 1)


def test_verify_sweep_adoption_criterion():
    records = [
        {"id": f"s{i}", "kind": "same_intent", "q1": "a", "q2": "b", "similarity": 0.9,
         "verify_same": i < 3} for i in range(4)
    ] + [
        {"id": f"d{i}", "kind": "different_intent", "q1": "a", "q2": "b", "similarity": 0.9,
         "verify_same": False} for i in range(4)
    ]
    rows = verify_sweep(records, [0.86, 0.95])
    assert _vrow(rows, 0.86)["adopt"] is True  # hit 75%, false 0%
    assert _vrow(rows, 0.95)["adopt"] is False  # hit 0%


def test_verify_sweep_missing_verify_field_is_miss():
    rows = verify_sweep(_records(), [0.86])  # 구버전 레코드(verify 없음)
    # 0.86 이상 후보는 검증 정보가 없으므로 miss, 직접 hit(>=0.95)만 인정
    assert _vrow(rows, 0.86)["same_hits"] == 2
    assert _vrow(rows, 0.86)["diff_hits"] == 1


def test_verify_stats():
    stats = verify_stats(_verify_records())
    assert stats["total"] == 8
    assert stats["false_yes_ids"] == ["d2"]
    assert stats["false_no_ids"] == ["s2"]
    assert stats["misjudged_ids"] == ["d2", "s2"]
    assert stats["accuracy"] == pytest.approx(6 / 8)
    assert stats["latency_p50_ms"] == pytest.approx(525.0)
    assert stats["avg_prompt_tokens"] == 40
    assert stats["outcomes"] == {"yes": 4, "no": 4}
    assert stats["models"] == ["haiku"] and stats["prompt_versions"] == ["v1"]


def test_verify_stats_without_verify_fields():
    assert verify_stats(_records()) == {"total": 0}


def test_render_verify_markdown():
    records = _verify_records()
    md = render_verify_markdown(records, verify_sweep(records), verify_stats(records))
    assert "## LLM 재검증 스윕" in md
    assert "| 0.86 |" in md
    assert "d2" in md and "s2" in md
    assert "채택" in md


def test_old_cache_report_has_no_verify_section():
    from scripts.eval.cache_eval import _report

    assert "LLM 재검증" not in _report(_records())
    assert "LLM 재검증 스윕" in _report(_verify_records())


# ---------------------------------------------------------------------------
# 검증 프롬프트 버전별 비교 (3회차)
# ---------------------------------------------------------------------------

from scripts.eval.cache_eval_metrics import (  # noqa: E402
    group_by_prompt_version,
    has_multiple_prompt_versions,
    parse_prompt_versions,
    render_version_comparison,
)


def _versioned_records() -> list[dict]:
    """v1: 후보 쌍을 전부 NO(과보수), v2: 정답과 일치하도록 판정한 합성 레코드."""
    base = _verify_records()
    v1 = [{**r, "verify_same": False, "verify_outcome": "no", "verify_prompt_version": "v1"} for r in base]
    v2 = [
        {**r, "verify_same": r["kind"] == "same_intent", "verify_outcome": "yes" if r["kind"] == "same_intent" else "no",
         "verify_prompt_version": "v2"}
        for r in base
    ]
    return v1 + v2


def test_parse_prompt_versions():
    assert parse_prompt_versions(None, "v1") == ["v1"]
    assert parse_prompt_versions("v2", "v1") == ["v2"]
    assert parse_prompt_versions("v1,v2", "v1") == ["v1", "v2"]
    assert parse_prompt_versions(" v1 , v2,v1,", "v1") == ["v1", "v2"]
    with pytest.raises(ValueError):
        parse_prompt_versions(" , ", "v1")


def test_group_by_prompt_version_and_old_cache():
    groups = group_by_prompt_version(_versioned_records())
    assert list(groups) == ["v1", "v2"]
    assert len(groups["v1"]) == len(groups["v2"]) == 8
    # 구버전 캐시(필드 없음)는 단일 그룹 "" 이고 다버전으로 취급하지 않는다
    old = _records()
    assert list(group_by_prompt_version(old)) == [""]
    assert has_multiple_prompt_versions(old) is False
    assert has_multiple_prompt_versions(_verify_records()) is False  # 단일 버전 v1
    assert has_multiple_prompt_versions(_versioned_records()) is True


def test_render_version_comparison_table():
    groups = group_by_prompt_version(_versioned_records())
    md = render_version_comparison(groups, direct_threshold=0.95)
    assert "## 검증 프롬프트 버전 비교" in md
    assert "hit_rate@0.80" in md and "hit_rate@0.85" in md and "hit_rate@0.86" in md
    assert "false_hit_rate" in md
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| v") and not line.startswith("| version")}
    assert set(rows) == {"v1", "v2"}
    # v1: 모든 쌍 NO -> false YES 0, false NO 4(same 4쌍)
    v1_cells = [c.strip() for c in rows["v1"].strip("|").split("|")]
    assert v1_cells[2:4] == ["0", "4"]
    # v2: 전부 정답 -> 정확도 100%, false YES/NO 0
    v2_cells = [c.strip() for c in rows["v2"].strip("|").split("|")]
    assert v2_cells[1] == "100.0%" and v2_cells[2:4] == ["0", "0"]


def test_report_multi_version_sections():
    from scripts.eval.cache_eval import _report

    report = _report(_versioned_records())
    assert "# 검증 프롬프트 v1" in report and "# 검증 프롬프트 v2" in report
    assert report.count("## LLM 재검증 스윕") == 2
    assert "## 검증 프롬프트 버전 비교" in report


def test_report_single_version_unchanged():
    from scripts.eval.cache_eval import _report

    report = _report(_verify_records())
    assert "검증 프롬프트 버전 비교" not in report
    assert report.count("## LLM 재검증 스윕") == 1


class TestHoldoutDatasetFile:
    def test_holdout_shape(self):
        pairs = load_dataset(_HOLDOUT)
        assert len(pairs) == 30
        assert sum(1 for p in pairs if p["kind"] == "same_intent") == 15
        assert sum(1 for p in pairs if p["kind"] == "different_intent") == 15
        ids = [p["id"] for p in pairs]
        assert len(set(ids)) == 30 and all(i.startswith("h") for i in ids)
        for p in pairs:
            assert p["q1"].strip() and p["q2"].strip() and p["q1"] != p["q2"]
            assert p["note"].strip()

    def test_holdout_has_no_sentence_overlap_with_tuning_set(self):
        tuning = {q for p in load_dataset(_DATASET) for q in (p["q1"], p["q2"])}
        holdout = [q for p in load_dataset(_HOLDOUT) for q in (p["q1"], p["q2"])]
        assert not (tuning & set(holdout))
        assert len(set(holdout)) == len(holdout)


def test_cli_verify_prompt_version_validation(capsys):
    from scripts.eval.cache_eval import main

    with pytest.raises(SystemExit):
        main(["--from-cache", "x.jsonl", "--verify-prompt-version", "v1"])  # --collect --verify 없이
    with pytest.raises(SystemExit):
        main(["--collect", "--verify", "--verify-prompt-version", "v1,v3"])
    assert "알 수 없는 검증 프롬프트 버전" in capsys.readouterr().err


# --- 임베딩 taskType (--task-type) ---------------------------------------------


def test_render_markdown_shows_task_type_only_when_given():
    records = _records()
    rows = sweep(records)
    with_tt = render_markdown(records, rows, {}, embedding_model="m", embedding_task_type="SEMANTIC_SIMILARITY")
    assert "- 임베딩 모델: m (taskType: SEMANTIC_SIMILARITY)" in with_tt
    without = render_markdown(records, rows, {}, embedding_model="m")
    assert "- 임베딩 모델: m\n" in without and "taskType" not in without


def test_cache_eval_cli_task_type_requires_collect(capsys):
    from scripts.eval import cache_eval

    with pytest.raises(SystemExit):
        cache_eval.main(["--task-type", "SEMANTIC_SIMILARITY"])
    assert "--task-type" in capsys.readouterr().err


def test_collect_records_stores_task_type(monkeypatch):
    import asyncio
    from types import SimpleNamespace

    from app.core.embeddings import embedder
    from scripts.eval import cache_eval

    calls = []

    async def fake_embed(texts, task_type=None):
        calls.append(task_type)
        return SimpleNamespace(embeddings=[[1.0, 0.0]] * len(texts), embedding_model="fake-model")

    monkeypatch.setattr(embedder, "embed_texts", fake_embed)

    records = asyncio.run(cache_eval._collect_records(_DATASET, 2, task_type="SEMANTIC_SIMILARITY"))
    assert calls == ["SEMANTIC_SIMILARITY"]
    assert {r["embedding_task_type"] for r in records} == {"SEMANTIC_SIMILARITY"}
    assert "(taskType: SEMANTIC_SIMILARITY)" in cache_eval._embedding_report(records)

    records = asyncio.run(cache_eval._collect_records(_DATASET, 2))
    assert calls[-1] is None
    assert {r["embedding_task_type"] for r in records} == {None}
    assert "taskType" not in cache_eval._embedding_report(records)


def test_old_records_without_task_type_render_as_before():
    from scripts.eval import cache_eval

    records = [{**r, "embedding_model": "old-model"} for r in _records()]
    report = cache_eval._embedding_report(records)
    assert "- 임베딩 모델: old-model\n" in report
    assert "taskType" not in report


# --- direct 임계치 스윕 / 항상 검증 (P1) -----------------------------------------


def _ds_records() -> list[dict]:
    # (id, kind, similarity, verify_same)
    rows = [
        ("s0", "same_intent", 0.97, False),   # direct 0.95에선 hit, 항상 검증에선 miss
        ("s1", "same_intent", 0.90, True),
        ("s2", "same_intent", 0.82, True),
        ("d0", "different_intent", 0.96, False),  # direct 0.95 false hit, 0.97/항상 검증 아님
        ("d1", "different_intent", 0.88, False),
        ("d2", "different_intent", 0.70, False),
    ]
    return [
        {"id": i, "kind": k, "q1": "a", "q2": "b", "similarity": s, "verify_same": v,
         "verify_outcome": "yes" if v else "no", "verify_prompt_version": "v1"}
        for i, k, s, v in rows
    ]


def _drow(rows, mode, direct, cand):
    return next(
        r for r in rows
        if r["mode"] == mode and r["direct_threshold"] == direct and r["candidate_threshold"] == cand
    )


def test_direct_sweep_false_hit_by_direct_shortcut():
    from scripts.eval.cache_eval_metrics import direct_sweep

    rows = direct_sweep(_ds_records())
    assert len(rows) == (4 + 1) * 2
    assert _drow(rows, "direct", 0.95, 0.80)["diff_hits"] == 1
    assert _drow(rows, "direct", 0.95, 0.80)["direct_hits_diff"] == 1
    assert _drow(rows, "direct", 0.97, 0.80)["diff_hits"] == 0
    assert _drow(rows, "always_verify", None, 0.80)["diff_hits"] == 0
    assert _drow(rows, "always_verify", None, 0.80)["direct_hits_diff"] == 0


def test_direct_sweep_always_verify_misses_direct_same_pair():
    from scripts.eval.cache_eval_metrics import direct_sweep

    rows = direct_sweep(_ds_records())
    assert _drow(rows, "direct", 0.95, 0.80)["same_hits"] == 3   # s0 즉시 + s1 + s2
    assert _drow(rows, "always_verify", None, 0.80)["same_hits"] == 2
    assert _drow(rows, "always_verify", None, 0.85)["same_hits"] == 1  # s2는 0.85 미만


def test_direct_sweep_verify_calls_and_options():
    from scripts.eval.cache_eval_metrics import direct_sweep

    rows = direct_sweep(_ds_records())
    av = _drow(rows, "always_verify", None, 0.80)
    # sim >= 0.80: s0 s1 s2 d0 d1 -> 5건 (same 3 / diff 2)
    assert (av["verify_calls"], av["verify_calls_same"], av["verify_calls_diff"]) == (5, 3, 2)
    d95 = _drow(rows, "direct", 0.95, 0.80)
    # 검증 대상: s1 s2 d1 (s0, d0는 즉시 hit)
    assert (d95["verify_calls"], d95["verify_calls_same"], d95["verify_calls_diff"]) == (3, 2, 1)
    only = direct_sweep(_ds_records(), [0.95], [0.80], include_always_verify=False)
    assert len(only) == 1 and only[0]["mode"] == "direct"


def test_direct_sweep_meets_l2_flag():
    from scripts.eval.cache_eval_metrics import direct_sweep

    rows = direct_sweep(_ds_records())
    # 항상 검증 @0.80: hit 2/3=67%, false 0% -> 충족
    assert _drow(rows, "always_verify", None, 0.80)["meets_l2"] is True
    # direct 0.95 @0.80: false 1/3 -> 불충족
    assert _drow(rows, "direct", 0.95, 0.80)["meets_l2"] is False
    # 항상 검증 @0.85: hit 1/3=33% < 40% -> 불충족
    assert _drow(rows, "always_verify", None, 0.85)["meets_l2"] is False


def test_render_direct_sweep_markdown():
    from scripts.eval.cache_eval_metrics import direct_sweep, render_direct_sweep_markdown

    md = render_direct_sweep_markdown(direct_sweep(_ds_records()))
    assert "## 즉시 hit 임계치 스윕 (direct / 항상 검증)" in md
    assert "| 항상 검증 | 0.80 |" in md
    assert "| direct 0.95 | 0.80 |" in md
    assert "L2 기준" in md


def _write_cache(path, records):
    from scripts.eval.cache_eval_metrics import save_cache

    save_cache(path, records)
    return str(path)


def test_cli_direct_sweep_requires_from_cache(capsys):
    from scripts.eval.cache_eval import main

    with pytest.raises(SystemExit):
        main(["--direct-sweep"])
    assert "--direct-sweep" in capsys.readouterr().err


def test_cli_direct_thresholds_requires_direct_sweep(capsys):
    from scripts.eval.cache_eval import main

    with pytest.raises(SystemExit):
        main(["--from-cache", "x.jsonl", "--direct-thresholds", "0.95"])
    assert "--direct-thresholds" in capsys.readouterr().err


def test_cli_direct_thresholds_validation(tmp_path, capsys):
    from scripts.eval.cache_eval import main

    path = _write_cache(tmp_path / "c.jsonl", _ds_records())
    for bad in ("abc", "0", "1.5"):
        with pytest.raises(SystemExit):
            main(["--from-cache", path, "--direct-sweep", "--direct-thresholds", bad])
    assert "--direct-thresholds" in capsys.readouterr().err


def test_cli_direct_sweep_end_to_end(tmp_path, capsys):
    from scripts.eval.cache_eval import main

    path = _write_cache(tmp_path / "c.jsonl", _ds_records())
    assert main(["--from-cache", path, "--direct-sweep", "--direct-thresholds", "0.95,0.97"]) == 0
    out = capsys.readouterr().out
    assert out.count("## 즉시 hit 임계치 스윕") == 1
    assert "direct 0.97" in out and "direct 0.96" not in out and "항상 검증" in out


def test_cli_direct_sweep_multi_version(tmp_path, capsys):
    from scripts.eval.cache_eval import main

    path = _write_cache(tmp_path / "c.jsonl", _versioned_records())
    assert main(["--from-cache", path, "--direct-sweep"]) == 0
    out = capsys.readouterr().out
    assert out.count("## 즉시 hit 임계치 스윕") == 2


def test_cli_direct_sweep_without_verify_fields(tmp_path, capsys):
    from scripts.eval.cache_eval import main

    path = _write_cache(tmp_path / "c.jsonl", _records())
    assert main(["--from-cache", path, "--direct-sweep"]) == 1
    captured = capsys.readouterr()
    assert "verify 필드" in captured.err and "즉시 hit 임계치 스윕" not in captured.out

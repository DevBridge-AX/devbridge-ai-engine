"""
scripts/metrics_report.py 집계 로직 유닛 테스트(C-4).

샘플 JSONL(성공/PARSE_WARN/단계별 실패 혼합)로 파싱 성공률, 단계별(*_ms) p50/p95,
실패 사유 분포가 기대대로 계산되는지, CLI(main())가 디렉터리 인자로 정상 종료 코드를
반환하는지 검증합니다. scripts/는 app 밖의 새 최상위 디렉터리이므로 sys.path에 repo
루트를 명시적으로 추가해 import합니다.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import metrics_report  # noqa: E402


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")


class TestLoadRecords:

    def test_groups_by_filename_and_skips_bad_lines(self, tmp_path):
        _write_jsonl(tmp_path / "ingestion.jsonl", [{"result": "COMPLETED"}])
        (tmp_path / "chat_metrics.jsonl").write_text(
            '{"event": "chat_metrics", "total_ms": 1.0}\nnot-json\n', encoding="utf-8"
        )

        records_by_event = metrics_report.load_records(tmp_path)

        assert set(records_by_event) == {"ingestion", "chat_metrics"}
        assert len(records_by_event["ingestion"]) == 1
        assert len(records_by_event["chat_metrics"]) == 1  # 깨진 줄은 건너뜀


class TestPercentileAndStages:

    def test_percentile_matches_known_values(self):
        values = [10.0, 20.0, 30.0, 40.0]
        assert metrics_report._percentile(values, 50) == 25.0
        assert metrics_report._percentile([], 50) == 0.0
        assert metrics_report._percentile([7.0], 95) == 7.0

    def test_stage_percentiles_only_numeric_ms_fields(self):
        records = [
            {"read_ms": 10.0, "embed_ms": None, "result": "COMPLETED"},
            {"read_ms": 20.0, "embed_ms": 5.0, "result": "COMPLETED"},
            {"read_ms": None, "note": "not a stage"},
        ]
        stages = metrics_report.compute_stage_percentiles(records)

        assert stages["read_ms"] == (15.0, 19.5)
        assert stages["embed_ms"] == (5.0, 5.0)
        assert "note" not in stages


class TestResultSummaryAndFailures:

    def test_success_rate_excludes_failed_only(self):
        records = [
            {"result": "COMPLETED"},
            {"result": "PARSE_WARN"},
            {"result": "EMPTY"},
            {"result": "FAILED", "failure_stage": "embed", "error_type": "RuntimeError"},
        ]
        summary = metrics_report.compute_result_summary(records)
        assert summary["total"] == 4
        assert summary["success_rate"] == 75.0
        assert summary["counts"]["FAILED"] == 1

    def test_no_result_field_returns_none(self):
        assert metrics_report.compute_result_summary([{"total_ms": 1.0}]) is None

    def test_failure_distribution_counts_stage_and_error_type(self):
        records = [
            {"result": "FAILED", "failure_stage": "embed", "error_type": "RuntimeError"},
            {"result": "FAILED", "failure_stage": "embed", "error_type": "TimeoutError"},
            {"result": "FAILED", "failure_stage": "read", "error_type": "FileNotFoundError"},
            {"result": "COMPLETED"},
        ]
        dist = metrics_report.compute_failure_distribution(records)
        assert dist["failure_stage"] == {"embed": 2, "read": 1}
        assert dist["error_type"] == {"RuntimeError": 1, "TimeoutError": 1, "FileNotFoundError": 1}


class TestLLMCallsBreakdown:

    def test_groups_by_purpose_and_model_with_stats(self):
        records = [
            {
                "purpose": "main", "model": "claude-a", "latency_ms": 100.0,
                "prompt_tokens": 50, "completion_tokens": 10, "parse_ok": True,
            },
            {
                "purpose": "main", "model": "claude-a", "latency_ms": 200.0,
                "prompt_tokens": 60, "completion_tokens": 20, "parse_ok": True,
            },
            {
                "purpose": "grounding", "model": "gemini-lite", "latency_ms": 50.0,
                "prompt_tokens": 120, "completion_tokens": 0, "parse_ok": False,
                "error_type": None,
            },
            {
                "purpose": "grounding", "model": "gemini-lite", "latency_ms": 999.0,
                "prompt_tokens": 0, "completion_tokens": 0, "parse_ok": True,
                "error_type": "HTTPStatusError",
            },
        ]

        breakdown = metrics_report.compute_llm_calls_breakdown(records)

        assert set(breakdown) == {("main", "claude-a"), ("grounding", "gemini-lite")}

        main_stats = breakdown[("main", "claude-a")]
        assert main_stats["count"] == 2
        assert main_stats["p50_latency_ms"] == 150.0
        assert main_stats["avg_prompt_tokens"] == 55.0
        assert main_stats["avg_completion_tokens"] == 15.0
        assert main_stats["parse_failure_rate"] == 0.0
        assert main_stats["error_count"] == 0

        grounding_stats = breakdown[("grounding", "gemini-lite")]
        assert grounding_stats["count"] == 2
        assert grounding_stats["parse_failure_rate"] == 50.0
        assert grounding_stats["error_count"] == 1

    def test_missing_purpose_or_model_grouped_as_unknown(self):
        breakdown = metrics_report.compute_llm_calls_breakdown([{"latency_ms": 10.0}])
        assert ("unknown", "unknown") in breakdown

    def test_empty_records_returns_empty_breakdown(self):
        assert metrics_report.compute_llm_calls_breakdown([]) == {}


class TestRenderMarkdownAndCli:

    def test_render_markdown_contains_event_sections(self):
        records_by_event = {
            "ingestion": [
                {"result": "COMPLETED", "read_ms": 5.0, "total_ms": 10.0},
                {"result": "FAILED", "failure_stage": "embed", "error_type": "RuntimeError", "total_ms": 3.0},
            ]
        }
        markdown = metrics_report.render_markdown(records_by_event)
        assert "## ingestion (2건)" in markdown
        assert "성공률" in markdown
        assert "read_ms" in markdown
        assert "embed=1" in markdown

    def test_render_markdown_handles_empty_dir(self):
        assert "집계할" in metrics_report.render_markdown({})

    def test_render_markdown_includes_llm_calls_breakdown_table(self):
        records_by_event = {
            "llm_calls": [
                {
                    "purpose": "main", "model": "claude-a", "latency_ms": 100.0,
                    "prompt_tokens": 50, "completion_tokens": 10, "parse_ok": True,
                },
                {
                    "purpose": "grounding", "model": "gemini-lite", "latency_ms": 50.0,
                    "prompt_tokens": 120, "completion_tokens": 0, "parse_ok": False,
                },
            ]
        }
        markdown = metrics_report.render_markdown(records_by_event)

        assert "## llm_calls (2건)" in markdown
        assert "purpose" in markdown and "model" in markdown
        assert "| main | claude-a |" in markdown
        assert "| grounding | gemini-lite |" in markdown

    def test_render_markdown_other_events_have_no_llm_calls_table(self):
        records_by_event = {"chat_metrics": [{"total_ms": 5.0}]}
        markdown = metrics_report.render_markdown(records_by_event)
        assert "평균 prompt_tokens" not in markdown

    def test_chat_metrics_cache_section_with_hits(self):
        records = [
            {"cache_enabled": True, "cache_hit": True, "total_ms": 10.0, "ttft_ms": 5.0},
            {"cache_enabled": True, "cache_hit": False, "total_ms": 2000.0, "ttft_ms": 800.0},
            {"cache_enabled": True, "cache_hit": False, "total_ms": 3000.0, "ttft_ms": 1000.0},
            {"cache_enabled": False, "cache_hit": False, "total_ms": 1.0, "ttft_ms": None},
        ]
        markdown = metrics_report.render_markdown({"chat_metrics": records})

        assert "시맨틱 캐시 hit율: 33.3% (1/3)" in markdown
        assert "| cache_hit=True | total_ms | 10.0 | 10.0 |" in markdown
        assert "| cache_hit=False | ttft_ms | 900.0 | 990.0 |" in markdown

    def test_chat_metrics_cache_section_na_when_disabled(self):
        records = [{"cache_enabled": False, "cache_hit": False, "total_ms": 5.0}, {"total_ms": 6.0}]
        markdown = metrics_report.render_markdown({"chat_metrics": records})

        assert "시맨틱 캐시 hit율: n/a" in markdown
        assert "cache_hit=" not in markdown

    def test_non_chat_events_have_no_cache_section(self):
        markdown = metrics_report.render_markdown({"ingestion": [{"result": "COMPLETED", "total_ms": 1.0}]})
        assert "시맨틱 캐시" not in markdown

    def test_main_returns_zero_for_valid_dir(self, tmp_path, capsys):
        _write_jsonl(tmp_path / "ingestion.jsonl", [{"result": "COMPLETED", "total_ms": 1.0}])

        exit_code = metrics_report.main([str(tmp_path)])

        assert exit_code == 0
        out = capsys.readouterr().out
        assert "ingestion" in out

    def test_main_returns_nonzero_for_missing_dir(self, tmp_path, capsys):
        missing = tmp_path / "does-not-exist"
        exit_code = metrics_report.main([str(missing)])
        assert exit_code == 1


class TestCacheVerifySummary:

    def _records(self):
        return [
            {"cache_enabled": True, "cache_hit": True, "cache_verify_result": "yes", "cache_verify_ms": 100.0},
            {"cache_enabled": True, "cache_hit": True, "cache_verify_result": None, "cache_verify_ms": None},
            {"cache_enabled": True, "cache_hit": False, "cache_verify_result": "no", "cache_verify_ms": 300.0},
            {"cache_enabled": True, "cache_hit": False, "cache_verify_result": "timeout", "cache_verify_ms": 500.0},
            {"cache_enabled": True, "cache_hit": False},
        ]

    def test_summary_keys(self):
        summary = metrics_report.compute_cache_summary(self._records())

        assert summary["verify_count"] == 3
        assert summary["verify_outcomes"] == {"yes": 1, "no": 1, "timeout": 1}
        assert summary["verify_latency"] == (300.0, 480.0)
        assert summary["verify_hit_count"] == 1
        assert summary["verify_hit_share"] == 50.0

    def test_render_lines(self):
        markdown = metrics_report.render_markdown({"chat_metrics": self._records()})

        assert "재검증 호출: 3 (yes 1 / no 1 / invalid 0 / error 0 / timeout 1)" in markdown
        assert "재검증 지연: p50 300.0ms / p95 480.0ms" in markdown
        assert "재검증 경유 hit 비율: 50.0% (1/2)" in markdown

    def test_zero_verify_records(self):
        records = [{"cache_enabled": True, "cache_hit": True}, {"cache_enabled": True, "cache_hit": False}]
        summary = metrics_report.compute_cache_summary(records)

        assert summary["verify_count"] == 0
        assert summary["verify_outcomes"] == {}
        assert summary["verify_latency"] is None
        assert summary["verify_hit_share"] == 0.0
        markdown = metrics_report.render_markdown({"chat_metrics": records})
        assert "재검증: n/a" in markdown
        assert "재검증 호출" not in markdown

    def test_no_hits_share_is_zero(self):
        records = [{"cache_enabled": True, "cache_hit": False, "cache_verify_result": "no", "cache_verify_ms": 1.0}]
        summary = metrics_report.compute_cache_summary(records)
        assert summary["verify_hit_share"] == 0.0

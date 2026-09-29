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

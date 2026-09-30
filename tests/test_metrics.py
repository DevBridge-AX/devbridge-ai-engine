"""
app/core/metrics.py 유닛 테스트.

StageTimer의 구간 측정과 record_metric의 JSONL 기록/비활성화/쓰기 실패 무시 동작을
검증합니다. metrics_dir는 항상 tmp_path로 돌려 레포에 파일이 남지 않도록 합니다.
"""

import json

import pytest
import time
from types import SimpleNamespace

from app.core import metrics


class TestStageTimer:

    def test_set_field_stores_generic_value(self):
        timer = metrics.StageTimer()
        timer.set_field("acl_refetch_count", 1)
        assert timer.fields == {"acl_refetch_count": 1}
        assert timer.stages == {}

    def test_start_stop_records_nonnegative_ms(self):
        timer = metrics.StageTimer()
        timer.start("stage_a")
        time.sleep(0.001)
        elapsed = timer.stop("stage_a")

        assert elapsed >= 0.0
        assert timer.stages["stage_a"] == elapsed

    def test_stop_without_start_is_noop(self):
        timer = metrics.StageTimer()
        assert timer.stop("never_started") == 0.0
        assert "never_started" not in timer.stages

    def test_measure_context_manager_records_stage(self):
        timer = metrics.StageTimer()
        with timer.measure("stage_b"):
            time.sleep(0.001)

        assert "stage_b" in timer.stages
        assert timer.stages["stage_b"] >= 0.0

    def test_measure_records_even_on_exception(self):
        timer = metrics.StageTimer()
        try:
            with timer.measure("stage_c"):
                raise ValueError("boom")
        except ValueError:
            pass

        assert "stage_c" in timer.stages


class TestRecordMetric:

    def _patch_settings(self, monkeypatch, **overrides):
        defaults = {"metrics_enabled": True, "metrics_dir": "./data/metrics"}
        defaults.update(overrides)
        monkeypatch.setattr(metrics, "get_settings", lambda: SimpleNamespace(**defaults))

    def test_writes_jsonl_line_to_metrics_dir(self, monkeypatch, tmp_path):
        self._patch_settings(monkeypatch, metrics_dir=str(tmp_path))

        metrics.record_metric("chat_metrics", {"total_ms": 12.5, "turn": 1})

        path = tmp_path / "chat_metrics.jsonl"
        assert path.exists()
        line = path.read_text(encoding="utf-8").strip()
        record = json.loads(line)
        assert record["event"] == "chat_metrics"
        assert record["total_ms"] == 12.5
        assert record["turn"] == 1

    def test_appends_multiple_records(self, monkeypatch, tmp_path):
        self._patch_settings(monkeypatch, metrics_dir=str(tmp_path))

        metrics.record_metric("ingestion", {"result": "COMPLETED"})
        metrics.record_metric("ingestion", {"result": "FAILED"})

        lines = (tmp_path / "ingestion.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_disabled_skips_file_write(self, monkeypatch, tmp_path):
        self._patch_settings(monkeypatch, metrics_enabled=False, metrics_dir=str(tmp_path))

        metrics.record_metric("chat_metrics", {"total_ms": 1.0})

        assert not (tmp_path / "chat_metrics.jsonl").exists()

    def test_file_write_failure_is_swallowed(self, monkeypatch, tmp_path, caplog):
        # metrics_dir 자리에 이미 동일 이름의 "파일"이 있으면 mkdir이 실패해 OSError가 발생한다.
        blocked = tmp_path / "blocked"
        blocked.write_text("not a directory", encoding="utf-8")
        self._patch_settings(monkeypatch, metrics_dir=str(blocked))

        caplog.set_level("WARNING", logger="devbridge.metrics")
        metrics.record_metric("chat_metrics", {"total_ms": 1.0})  # 예외가 전파되지 않아야 함

        assert any("metrics 파일 기록 실패" in r.message for r in caplog.records)

    def test_no_pii_leak_helper_shape(self, monkeypatch, tmp_path):
        """호출부 책임이지만, 정상 payload가 그대로 직렬화되는지 최소 확인."""
        self._patch_settings(monkeypatch, metrics_dir=str(tmp_path))

        metrics.record_metric(
            "chat_metrics", {"session_id_hash": "abc123def456", "turn": 2}
        )

        record = json.loads((tmp_path / "chat_metrics.jsonl").read_text(encoding="utf-8").strip())
        assert record["session_id_hash"] == "abc123def456"
        assert "question" not in record
        assert "user_id" not in record


def test_stage_timer_accumulates_repeated_stage(monkeypatch):
    """같은 구간을 두 번 측정하면 합산되어야 한다(ACL 재조회 시 vector_ms 등)."""
    ticks = iter([0.0, 0.010, 1.0, 1.005])  # 10ms, 5ms
    monkeypatch.setattr(metrics.time, "perf_counter", lambda: next(ticks))
    timer = metrics.StageTimer()
    with timer.measure("vector_ms"):
        pass
    with timer.measure("vector_ms"):
        pass
    assert timer.stages["vector_ms"] == pytest.approx(15.0)

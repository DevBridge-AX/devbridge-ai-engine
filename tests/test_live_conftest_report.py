"""
tests/live/conftest.py::pytest_sessionfinish 집계 로직 단위 테스트(non-live, API 호출 없음).

_LIVE_STATE를 직접 채운 뒤 훅을 호출해 llm_calls/chat_metrics jsonl이 purpose×model별
토큰 합계/지연 분위수로 집계되어 data/live_runs/*.json에 저장되는지 확인합니다.
live_env가 셋업되지 않은 세션(_LIVE_STATE 비어있음)에서는 아무 것도 하지 않는지도 확인합니다.
"""

import json

from tests.live import conftest as live_conftest


def _write_jsonl(path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8"
    )


def test_sessionfinish_is_noop_when_live_not_run(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(live_conftest, "_LIVE_STATE", {})
    monkeypatch.setattr(live_conftest, "_REPO_ROOT", tmp_path)

    live_conftest.pytest_sessionfinish(session=None, exitstatus=0)

    assert not (tmp_path / "data" / "live_runs").exists()
    assert capsys.readouterr().out == ""


def test_sessionfinish_aggregates_tokens_and_writes_report(monkeypatch, tmp_path, capsys):
    metrics_dir = tmp_path / "metrics"
    metrics_dir.mkdir()
    _write_jsonl(
        metrics_dir / "llm_calls.jsonl",
        [
            {
                "purpose": "main_stream",
                "model": "claude-x",
                "latency_ms": 100.0,
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "thoughts_tokens": 0,
                "parse_ok": True,
            },
            {
                "purpose": "rewrite",
                "model": "gemini-x",
                "latency_ms": 50.0,
                "prompt_tokens": 30,
                "completion_tokens": 10,
                "thoughts_tokens": 0,
                "parse_ok": True,
            },
        ],
    )
    _write_jsonl(
        metrics_dir / "chat_metrics.jsonl",
        [{"total_ms": 500.0, "llm_ms": 300.0}],
    )

    fake_state = {
        "metrics_dir": metrics_dir,
        "workspace_id": "live-testws",
        "persona_answers": {"planner": "안녕하세요"},
    }
    monkeypatch.setattr(live_conftest, "_LIVE_STATE", fake_state)
    monkeypatch.setattr(live_conftest, "_REPO_ROOT", tmp_path)

    live_conftest.pytest_sessionfinish(session=None, exitstatus=0)

    out_files = list((tmp_path / "data" / "live_runs").glob("*.json"))
    assert len(out_files) == 1

    report = json.loads(out_files[0].read_text(encoding="utf-8"))
    assert report["workspace_id"] == "live-testws"
    assert report["llm_calls_count"] == 2
    assert report["chat_requests_count"] == 1
    assert report["token_sums_by_purpose_model"]["main_stream×claude-x"] == {
        "prompt_tokens": 50,
        "completion_tokens": 20,
        "thoughts_tokens": 0,
    }
    assert report["token_sums_by_purpose_model"]["rewrite×gemini-x"]["completion_tokens"] == 10
    assert "main_stream×claude-x" in report["llm_calls_latency"]
    assert "total_ms" in report["chat_stage_latency_ms"]
    assert report["persona_answers_preview"] == {"planner": "안녕하세요"}

    assert "라이브 검증 토큰/지연 집계" in capsys.readouterr().out

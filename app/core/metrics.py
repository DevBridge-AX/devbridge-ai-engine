"""
공통 관측성(observability) 계측 모듈.

StageTimer: perf_counter 기반 구간별 소요 시간(ms) 측정 헬퍼.
record_metric(): 이벤트 1건을 구조화 로그(logger="devbridge.metrics") 1줄 +
  `{metrics_dir}/{event}.jsonl` append로 기록합니다.

주의:
- payload에는 질문/답변 원문, user_id 등 PII를 절대 포함하지 않습니다(호출부 책임).
  session_id는 호출부에서 SHA-256 해시 앞 12자리로 변환해 전달해야 합니다.
- metrics_dir 파일 쓰기 실패는 요청/인덱싱 흐름에 영향을 주지 않도록 warning 로그만
  남기고 무시합니다. metrics_enabled=False면 로그만 남기고 파일에는 쓰지 않습니다.
"""

import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger("devbridge.metrics")


class StageTimer:
    """구간별 소요 시간(ms)을 기록합니다. 동일 이름은 마지막 측정값으로 덮어씁니다."""

    def __init__(self) -> None:
        self._starts: dict[str, float] = {}
        self.stages: dict[str, float] = {}
        self.fields: dict[str, object] = {}

    def start(self, name: str) -> None:
        """name 구간의 시작 시각을 기록합니다."""
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        """start(name) 이후 경과 시간(ms)을 stages에 기록하고 반환합니다.

        start 없이 호출되면 아무 것도 기록하지 않고 0.0을 반환합니다.
        """
        started = self._starts.pop(name, None)
        if started is None:
            return 0.0

        elapsed_ms = (time.perf_counter() - started) * 1000
        # 같은 구간이 재실행되면(예: ACL 재조회) 덮어쓰지 않고 누적한다.
        self.stages[name] = self.stages.get(name, 0.0) + elapsed_ms
        return elapsed_ms

    def set_field(self, name: str, value: object) -> None:
        """ms 구간이 아닌 부가 계측값(카운트 등)을 기록합니다."""
        self.fields[name] = value

    @contextmanager
    def measure(self, name: str):
        """`with timer.measure("name"):` 형태로 구간을 측정합니다."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)


def record_metric(event: str, payload: dict) -> None:
    """metrics 이벤트 1건을 로그 + JSONL 파일로 기록합니다.

    payload는 질문/답변 원문·user_id를 포함해서는 안 됩니다(호출부 책임).
    파일 쓰기 실패는 요청/인덱싱 흐름을 막지 않도록 경고 로그만 남기고 무시합니다.
    """
    record = {"event": event, **payload}
    line = json.dumps(record, ensure_ascii=False, default=str)
    logger.info(line)

    settings = get_settings()
    if not getattr(settings, "metrics_enabled", True):
        return

    try:
        metrics_dir = Path(settings.metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        path = metrics_dir / f"{event}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        logger.warning("record_metric: metrics 파일 기록 실패 (event=%s)", event, exc_info=True)

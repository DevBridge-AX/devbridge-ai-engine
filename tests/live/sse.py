"""
SSE(text/event-stream) 응답 파서.

TestClient가 수집한 /chat 응답 본문("event: ...\\ndata: ...\\n\\n" 반복)을
(event, data) 튜플 리스트로 변환합니다. tests/live의 라이브 테스트와
tests/test_live_sse_parser.py(단위 테스트)에서 사용합니다.
"""

import json


def parse_sse(text: str) -> list[tuple[str, dict]]:
    """SSE 텍스트를 [(event, data_dict), ...]로 파싱합니다.

    이벤트 블록은 빈 줄(\\n\\n)로 구분되며, 각 블록은 "event: {name}"과
    "data: {json}" 줄을 포함합니다. data 줄이 여러 개면 이어붙여 파싱합니다.
    JSON 파싱에 실패하면 해당 이벤트의 data는 빈 dict로 반환합니다.
    """
    events: list[tuple[str, dict]] = []
    blocks = text.replace("\r\n", "\n").split("\n\n")

    for block in blocks:
        block = block.strip("\n")
        if not block:
            continue

        event_name: str | None = None
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_name = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_lines.append(line[len("data:"):].strip())

        if event_name is None:
            continue

        data_text = "\n".join(data_lines)
        try:
            data = json.loads(data_text) if data_text else {}
        except json.JSONDecodeError:
            data = {}

        events.append((event_name, data))

    return events

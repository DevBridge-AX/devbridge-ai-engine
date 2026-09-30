"""
tests/live/sse.py::parse_sse 단위 테스트. API 호출 없이 동작합니다(non-live).
"""

from tests.live.sse import parse_sse


def test_parses_token_and_done_events_in_order():
    text = (
        'event: token\ndata: {"text": "안녕"}\n\n'
        'event: token\ndata: {"text": "하세요"}\n\n'
        'event: done\ndata: {"is_groundable": true}\n\n'
    )

    events = parse_sse(text)

    assert [name for name, _ in events] == ["token", "token", "done"]
    assert events[0][1] == {"text": "안녕"}
    assert events[1][1] == {"text": "하세요"}
    assert events[2][1] == {"is_groundable": True}


def test_ignores_trailing_whitespace_and_empty_blocks():
    text = 'event: done\ndata: {"a": 1}\n\n\n'

    events = parse_sse(text)

    assert events == [("done", {"a": 1})]


def test_returns_empty_list_for_empty_text():
    assert parse_sse("") == []


def test_multiline_data_is_joined_before_json_parse():
    text = 'event: done\ndata: {"a": 1,\ndata: "b": 2}\n\n'

    events = parse_sse(text)

    assert events == [("done", {"a": 1, "b": 2})]


def test_block_without_event_line_is_skipped():
    text = 'data: {"a": 1}\n\nevent: done\ndata: {"b": 2}\n\n'

    events = parse_sse(text)

    assert events == [("done", {"b": 2})]

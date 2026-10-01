"""app/core/llm/rewrite_guard.py 유닛 테스트 (상세 케이스는 test_rewrite_eval_metrics.py 참고)."""

import pytest

from app.core.llm.rewrite_guard import is_answer_like


@pytest.mark.parametrize(
    "rewritten",
    [
        "가" * 81,
        "첫 줄\n둘째 줄",
        "결제 API 타임아웃은 30초입니다.",
        '재작성된 "쿼리"',
        "출력: 결제 API 타임아웃",
    ],
)
def test_answer_like_true(rewritten):
    assert is_answer_like("질문입니다 정말로", rewritten) is True


def test_normal_query_is_not_answer_like():
    assert is_answer_like("그거 타임아웃 몇 초야?", "결제 API 타임아웃 시간") is False

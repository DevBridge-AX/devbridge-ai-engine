"""app/core/llm/rewrite_guard.py 유닛 테스트 (상세 케이스는 test_rewrite_eval_metrics.py 참고)."""

import pytest

from app.core.llm.rewrite_guard import is_answer_like


@pytest.mark.parametrize(
    "rewritten",
    [
        "가" * 81,
        "첫 줄\n둘째 줄",
        "결제 API 타임아웃은 30초입니다.",
        '"재작성된 쿼리"',
        "출력: 결제 API 타임아웃",
    ],
)
def test_answer_like_true(rewritten):
    assert is_answer_like("질문입니다 정말로", rewritten) is True


def test_normal_query_is_not_answer_like():
    assert is_answer_like("그거 타임아웃 몇 초야?", "결제 API 타임아웃 시간") is False


def test_short_followup_expanded_within_floor_is_not_answer_like():
    """짧은 후속 질문을 맥락 복원한 결과(40자 이하)는 4배를 넘어도 답변형이 아니다."""
    assert is_answer_like("그거 어디?", "결제 API 소스 코드 위치") is False


def test_output_identical_to_original_is_not_answer_like():
    """잡담 passthrough(출력 == 원문)는 서술형 종결이어도 답변형이 아니다."""
    assert is_answer_like("감사합니다.", " 감사합니다. ") is False


def test_partial_quote_in_query_is_not_answer_like():
    assert is_answer_like("그거 소스 어디?", "결제 API의 'cancel' 엔드포인트 소스 코드 위치") is False


def test_short_followup_over_floor_is_answer_like():
    assert is_answer_like("왜?", "결제 API는 TossPayments와 연동되어 카드 및 계좌이체 결제를 처리하는 모듈") is True

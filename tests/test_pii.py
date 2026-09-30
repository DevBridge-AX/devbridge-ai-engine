"""PII 스크러빙 유닛 테스트."""

import pytest

from app.core.utils.pii import scrub_text


@pytest.mark.parametrize(
    "text,expected",
    [
        ("문의: hong.gd+a@example.co.kr 로", "문의: [EMAIL] 로"),
        ("전화 010-1234-5678 입니다", "전화 [PHONE] 입니다"),
        ("전화 01012345678", "전화 [PHONE]"),
        ("사무실 02-123-4567", "사무실 [PHONE]"),
        ("사무실 031-1234-5678", "사무실 [PHONE]"),
        ("해외 +82-10-1234-5678", "해외 [PHONE]"),
        ("해외 +82 2 123 4567", "해외 [PHONE]"),
        ("주민번호 900101-1234567", "주민번호 [RRN]"),
        ("서버 192.168.0.1 접속", "서버 [IP] 접속"),
        ("주소 https://example.com/a?b=1 참고", "주소 [URL] 참고"),
    ],
)
def test_each_pattern(text, expected):
    scrubbed, count = scrub_text(text)
    assert scrubbed == expected
    assert count == 1


def test_rrn_does_not_swallow_phone_and_vice_versa():
    scrubbed, count = scrub_text("010-1234-5678 / 900101-1234567")
    assert scrubbed == "[PHONE] / [RRN]"
    assert count == 2


def test_url_with_ip_and_email_counted_once():
    scrubbed, count = scrub_text("http://10.0.0.1:8080/x?u=a@b.com")
    assert scrubbed == "[URL]"
    assert count == 1


def test_multiple_and_deterministic():
    text = "a@b.com, c@d.com 그리고 010-1111-2222"
    assert scrub_text(text) == scrub_text(text) == ("[EMAIL], [EMAIL] 그리고 [PHONE]", 3)


def test_no_false_positive_on_normal_korean_sentence():
    text = "2026년 9월 30일 3건의 이슈가 등록되었고 v1.2.3 버전은 2026.09.30 배포, 총 1,234,567원입니다."
    assert scrub_text(text) == (text, 0)

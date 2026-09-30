"""
정규식 기반 PII 스크러빙.

치환 토큰: [URL] [EMAIL] [RRN] [PHONE] [IP]

주의: 한국어 이름(인명) 탐지는 의도적으로 하지 않는다. 일반 명사/고유명사와의
오탐이 너무 많아 학습 데이터 품질을 해치기 때문이다.

치환 순서는 고정이다(URL -> EMAIL -> RRN -> PHONE -> IP).
- URL을 먼저 치환해 URL 내부의 IP/이메일 형태 문자열이 이중 처리되지 않게 한다.
- RRN을 PHONE보다 먼저 치환한다. RRN은 생년월일 유효성(월/일)과 성별 자리(1-4)를
  요구해 전화번호와 겹치지 않도록 좁게 정의한다.
"""

import re

_URL = re.compile(r"https?://[^\s<>\"'\)\]]+", re.IGNORECASE)
_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+")
_RRN = re.compile(
    r"(?<!\d)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])-?[1-4]\d{6}(?!\d)"
)
_PHONE = re.compile(
    r"(?<![\d-])(?:"
    r"\+82[-.\s]?(?:0?1[016789]|0?2|0?[3-6]\d|0?70)[-.\s]?\d{3,4}[-.\s]?\d{4}"
    r"|01[016789][-.\s]?\d{3,4}[-.\s]?\d{4}"
    r"|0(?:2|[3-6]\d|70)[-.\s]?\d{3,4}[-.\s]?\d{4}"
    r")(?![\d-])"
)
_IPV4 = re.compile(
    r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}"
    r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?!\d)(?!\.\d)"
)

_RULES: tuple[tuple[re.Pattern, str], ...] = (
    (_URL, "[URL]"),
    (_EMAIL, "[EMAIL]"),
    (_RRN, "[RRN]"),
    (_PHONE, "[PHONE]"),
    (_IPV4, "[IP]"),
)


def scrub_text(text: str) -> tuple[str, int]:
    """PII를 고정 토큰으로 치환하고 (치환된 텍스트, 치환 횟수)를 반환한다."""
    total = 0
    for pattern, token in _RULES:
        text, n = pattern.subn(token, text)
        total += n
    return text, total

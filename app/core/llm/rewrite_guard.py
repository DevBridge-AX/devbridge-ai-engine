"""
rewrite 결과 가드용 순수 휴리스틱.

API 호출이나 DB 접근 없이 rewrite 출력이 '쿼리'가 아니라 '답변'처럼 보이는지 판별합니다.
"""

_ANSWER_LIKE_ENDINGS = ("입니다.", "습니다.")
_ANSWER_LIKE_PREFIXES = ("출력:", "재작성:")
_QUOTE_CHARS = ('"', "'", "“", "”", "‘", "’", "「", "」")


def is_answer_like(original: str, rewritten: str) -> bool:
    """rewrite 결과가 재작성된 '쿼리'가 아니라 질문에 대한 '답변'처럼 보이는지 판별합니다.

    아래 중 하나라도 해당하면 답변형(True)으로 판정합니다.
    - 길이가 원래 질문의 4배 초과 또는 80자 초과
    - 줄바꿈 포함
    - "~입니다." 또는 "~습니다."로 끝남
    - 따옴표류 문자 포함
    - "출력:" 또는 "재작성:" 접두어로 시작
    """
    stripped = rewritten.strip()
    if not stripped:
        return False

    if len(stripped) > len(original) * 4 or len(stripped) > 80:
        return True

    if "\n" in rewritten:
        return True

    if stripped.endswith(_ANSWER_LIKE_ENDINGS):
        return True

    if any(q in stripped for q in _QUOTE_CHARS):
        return True

    if stripped.startswith(_ANSWER_LIKE_PREFIXES):
        return True

    return False

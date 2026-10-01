"""
rewrite 결과 가드용 순수 휴리스틱.

API 호출이나 DB 접근 없이 rewrite 출력이 '쿼리'가 아니라 '답변'처럼 보이는지 판별합니다.
"""

_ANSWER_LIKE_ENDINGS = ("니다.", "니다")  # 입니다/습니다/합니다/됩니다 등 서술형 종결
_ANSWER_LIKE_PREFIXES = ("출력:", "재작성:")
_QUOTE_CHARS = ('"', "'", "“", "”", "‘", "’", "「", "」")
_MIN_LENGTH_FLOOR = 40  # 원문이 아주 짧을 때 4배 규칙 대신 적용하는 길이 하한
_MAX_LENGTH = 80


def is_answer_like(original: str, rewritten: str) -> bool:
    """rewrite 결과가 재작성된 '쿼리'가 아니라 질문에 대한 '답변'처럼 보이는지 판별합니다.

    출력이 원문과 (strip 후) 같으면 잡담 passthrough이므로 항상 False입니다.
    그 외에 아래 중 하나라도 해당하면 답변형(True)으로 판정합니다.
    - 길이가 원래 질문의 4배 초과(단, 하한 40자 — 짧은 후속 질문("그거 어디?")을 맥락
      복원한 정상 결과를 답변형으로 오판하지 않도록) 또는 80자 초과
    - 줄바꿈 포함
    - "~니다."로 끝남(입니다/습니다/합니다/됩니다 등 서술형 종결)
    - 출력 전체가 따옴표류로 감싸져 있음(쿼리 중간의 부분 따옴표는 정상으로 본다)
    - "출력:" 또는 "재작성:" 접두어로 시작
    """
    stripped = rewritten.strip()
    if not stripped:
        return False

    if stripped == original.strip():
        return False

    if len(stripped) > max(len(original) * 4, _MIN_LENGTH_FLOOR) or len(stripped) > _MAX_LENGTH:
        return True

    if "\n" in rewritten:
        return True

    if stripped.endswith(_ANSWER_LIKE_ENDINGS):
        return True

    if stripped[0] in _QUOTE_CHARS and stripped[-1] in _QUOTE_CHARS:
        return True

    if stripped.startswith(_ANSWER_LIKE_PREFIXES):
        return True

    return False

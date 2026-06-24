"""
BM25용 텍스트 토크나이저.

Korean + English 혼합 텍스트를 BM25 입력 토큰 리스트로 변환합니다.
현재 구현: regex 기반 (외부 의존성 없음).
향후 업그레이드: konlpy/mecab 기반 형태소 분석기로 교체 가능 (Tokenizer Protocol 준수).
"""

import re
from typing import Protocol


class Tokenizer(Protocol):
    def __call__(self, text: str) -> list[str]: ...


_TOKEN_PATTERN = re.compile(r"[가-힣]+|[a-zA-Z0-9_]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """텍스트를 BM25용 토큰 리스트로 변환합니다.

    - 한글 연속 → 토큰 (예: "인증흐름" → ["인증흐름"])
    - 4자 이상 한글 → bigram 추가 (복합어 recall 향상)
    - 영문 → 소문자 변환
    - 구두점/특수문자 → 자연 탈락
    """
    raw_tokens = _TOKEN_PATTERN.findall(text)
    result: list[str] = []
    for token in raw_tokens:
        lower = token.lower()
        result.append(lower)
        if len(token) >= 4 and _is_korean(token):
            for i in range(len(token) - 1):
                result.append(token[i : i + 2])
    return result


def _is_korean(s: str) -> bool:
    return all("가" <= c <= "힣" for c in s)

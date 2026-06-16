"""
[문서 청킹] 실험 파트

슬라이딩 윈도우(고정 크기 + 오버랩) 방식으로 문서를 청크 분할
doc_type에 따라 텍스트/코드 두 전략 사용

청크 크기 근거:
- 텍스트(TEXT_CHUNK_SIZE=1500 chars ≈ 375 tokens): 충분한 문맥을 제공하면서
  text-embedding-3-large의 8191 토큰 한도에 여유
- 코드/diff(CODE_CHUNK_SIZE=800 chars ≈ 200 tokens): 함수/블록 단위 지역성을
  유지하기 위해 작게 설정
- 오버랩은 각 크기의 약 13%로 청크 경계에서의 컨텍스트 단절을 완화
"""

from dataclasses import dataclass

TEXT_CHUNK_SIZE = 1500
TEXT_OVERLAP = 200

CODE_CHUNK_SIZE = 800
CODE_OVERLAP = 100

_CODE_TYPES = frozenset(
    {
        "python", "javascript", "typescript", "java", "kotlin",
        "go", "rust", "cpp", "c", "git_diff",
    }
)


@dataclass
class Chunk:
    content: str
    chunk_metadata: dict


def chunk_document(text: str, doc_type: str = "text") -> list[Chunk]:
    """텍스트를 doc_type에 맞는 전략으로 청킹합니다.

    반환값: content(원문)와 chunk_metadata(chunk_index, char_start, char_end)를
    담은 Chunk 리스트.
    """
    text = text.strip()
    if not text:
        return []

    if doc_type in _CODE_TYPES:
        return _sliding_window(text, CODE_CHUNK_SIZE, CODE_OVERLAP)
    return _sliding_window(text, TEXT_CHUNK_SIZE, TEXT_OVERLAP)


def _sliding_window(text: str, size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(
            Chunk(
                content=text[start:end],
                chunk_metadata={"chunk_index": idx, "char_start": start, "char_end": end},
            )
        )
        if end == len(text):
            break
        start += size - overlap
        idx += 1
    return chunks

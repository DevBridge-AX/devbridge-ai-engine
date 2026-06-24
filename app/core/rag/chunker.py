"""
시맨틱 청킹 모듈.

doc_type별 의미 단위 분할 전략:
  - markdown(md): 헤더 경계 → 단락 → 슬라이딩 윈도우 fallback
  - code(python, java, ts 등): 함수/클래스 경계 → 슬라이딩 윈도우 fallback
  - sql: 문(statement) 단위 → 슬라이딩 윈도우 fallback
  - git_diff: 파일 단위 diff 경계 → 슬라이딩 윈도우 fallback
  - text/기타: 단락(\n\n) → 문장 → 슬라이딩 윈도우 fallback

각 전략에서 개별 세그먼트가 max size를 초과하면 슬라이딩 윈도우로 fallback.
"""

import re
from dataclasses import dataclass

TEXT_CHUNK_SIZE = 1500
TEXT_OVERLAP = 200

CODE_CHUNK_SIZE = 800
CODE_OVERLAP = 100

_CODE_TYPES = frozenset(
    {
        "python", "javascript", "typescript", "java", "kotlin",
        "go", "rust", "cpp", "c",
    }
)

_MARKDOWN_TYPES = frozenset({"markdown", "md"})

_SQL_TYPES = frozenset({"sql"})


@dataclass
class Chunk:
    content: str
    chunk_metadata: dict


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_document(text: str, doc_type: str = "text") -> list[Chunk]:
    """텍스트를 doc_type에 맞는 전략으로 청킹합니다.

    반환값: content(원문)와 chunk_metadata(chunk_index, char_start, char_end)를
    담은 Chunk 리스트.
    """
    text = text.strip()
    if not text:
        return []

    if doc_type in _MARKDOWN_TYPES:
        segments = _split_markdown(text)
        return _segments_to_chunks(text, segments, TEXT_CHUNK_SIZE, TEXT_OVERLAP)

    if doc_type in _SQL_TYPES:
        segments = _split_sql(text)
        return _segments_to_chunks(text, segments, CODE_CHUNK_SIZE, CODE_OVERLAP)

    if doc_type == "git_diff":
        segments = _split_git_diff(text)
        return _segments_to_chunks(text, segments, CODE_CHUNK_SIZE, CODE_OVERLAP)

    if doc_type in _CODE_TYPES:
        segments = _split_code(text, doc_type)
        return _segments_to_chunks(text, segments, CODE_CHUNK_SIZE, CODE_OVERLAP)

    segments = _split_text(text)
    return _segments_to_chunks(text, segments, TEXT_CHUNK_SIZE, TEXT_OVERLAP)


# ---------------------------------------------------------------------------
# 슬라이딩 윈도우 (fallback)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 세그먼트 → 청크 변환 (공통)
# ---------------------------------------------------------------------------

def _segments_to_chunks(
    full_text: str,
    segments: list[str],
    max_size: int,
    overlap: int,
) -> list[Chunk]:
    """의미 단위 세그먼트를 청크로 변환합니다.

    - 세그먼트가 max_size 이하면 그대로 1청크
    - 초과하면 슬라이딩 윈도우로 분할
    - 작은 세그먼트는 인접 세그먼트와 병합하여 너무 잘게 쪼개지는 것을 방지
    """
    if not segments:
        return _sliding_window(full_text, max_size, overlap)

    merged = _merge_small_segments(segments, max_size)

    chunks: list[Chunk] = []
    offset = 0
    idx = 0

    for segment in merged:
        seg_start = full_text.find(segment, offset)
        if seg_start == -1:
            seg_start = offset

        if len(segment) <= max_size:
            chunks.append(Chunk(
                content=segment,
                chunk_metadata={
                    "chunk_index": idx,
                    "char_start": seg_start,
                    "char_end": seg_start + len(segment),
                },
            ))
            idx += 1
        else:
            sub_chunks = _sliding_window(segment, max_size, overlap)
            for sc in sub_chunks:
                sc.chunk_metadata["chunk_index"] = idx
                sc.chunk_metadata["char_start"] += seg_start
                sc.chunk_metadata["char_end"] += seg_start
                chunks.append(sc)
                idx += 1

        offset = seg_start + len(segment)

    return chunks if chunks else _sliding_window(full_text, max_size, overlap)


def _merge_small_segments(segments: list[str], max_size: int) -> list[str]:
    """max_size의 20% 미만인 세그먼트를 인접 세그먼트와 병합합니다.

    병합 후 결과가 max_size를 초과하면 병합하지 않습니다.
    """
    threshold = max_size * 0.2
    merged: list[str] = []

    for seg in segments:
        if merged and len(seg) < threshold and len(merged[-1]) + len(seg) + 1 <= max_size:
            merged[-1] = merged[-1] + "\n" + seg
        elif merged and len(merged[-1]) < threshold and len(merged[-1]) + len(seg) + 1 <= max_size:
            merged[-1] = merged[-1] + "\n" + seg
        else:
            merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# 마크다운 분할
# ---------------------------------------------------------------------------

_MD_HEADER_RE = re.compile(r"^(#{1,4})\s+", re.MULTILINE)


def _split_markdown(text: str) -> list[str]:
    """마크다운 헤더 경계로 섹션 분할, 대형 섹션은 단락 하위 분할."""
    parts = _MD_HEADER_RE.split(text)

    sections: list[str] = []
    i = 0
    while i < len(parts):
        if _MD_HEADER_RE.match(parts[i] if parts[i] else ""):
            if i + 1 < len(parts):
                sections.append(parts[i] + parts[i + 1])
                i += 2
            else:
                sections.append(parts[i])
                i += 1
        else:
            if parts[i].strip():
                sections.append(parts[i])
            i += 1

    result: list[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= TEXT_CHUNK_SIZE:
            result.append(section)
        else:
            paragraphs = _split_paragraphs(section)
            result.extend(paragraphs)

    return result


# ---------------------------------------------------------------------------
# 코드 분할
# ---------------------------------------------------------------------------

_PYTHON_BOUNDARY_RE = re.compile(
    r"^(?=(?:def |class |async def ))", re.MULTILINE
)

_C_STYLE_BOUNDARY_RE = re.compile(
    r"^(?=\s*(?:(?:public|private|protected|static|async|export|function|const|let|var)\s+)*"
    r"(?:(?:fun|func|fn|def|class|interface|enum|struct|impl)\s|\w+\s+\w+\s*\())",
    re.MULTILINE,
)


def _split_code(text: str, doc_type: str) -> list[str]:
    """함수/클래스 경계 기반으로 코드를 분할합니다."""
    if doc_type == "python":
        pattern = _PYTHON_BOUNDARY_RE
    else:
        pattern = _C_STYLE_BOUNDARY_RE

    parts = pattern.split(text)
    segments = [p for p in parts if p.strip()]

    if len(segments) <= 1:
        return _split_by_blank_lines(text)

    return segments


def _split_by_blank_lines(text: str) -> list[str]:
    """빈 줄(2줄 이상 연속) 기준으로 분할합니다."""
    parts = re.split(r"\n{2,}", text)
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# SQL 분할
# ---------------------------------------------------------------------------

_SQL_STATEMENT_RE = re.compile(
    r"(?:;\s*\n)",
)


def _split_sql(text: str) -> list[str]:
    """SQL 문(statement) 단위로 분할합니다."""
    parts = _SQL_STATEMENT_RE.split(text)
    segments: list[str] = []

    for part in parts:
        part = part.strip()
        if part:
            if not part.endswith(";"):
                part += ";"
            segments.append(part)

    return segments if len(segments) > 1 else [text]


# ---------------------------------------------------------------------------
# Git diff 분할
# ---------------------------------------------------------------------------

_DIFF_FILE_RE = re.compile(r"^(?=diff --git )", re.MULTILINE)


def _split_git_diff(text: str) -> list[str]:
    """diff --git 파일 경계 기준으로 분할합니다."""
    parts = _DIFF_FILE_RE.split(text)
    segments = [p for p in parts if p.strip()]
    return segments if len(segments) > 1 else [text]


# ---------------------------------------------------------------------------
# 일반 텍스트 분할
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"(?<=[.!?。])\s+")


def _split_text(text: str) -> list[str]:
    """단락 → 문장 순으로 분할합니다."""
    paragraphs = _split_paragraphs(text)

    result: list[str] = []
    for para in paragraphs:
        if len(para) <= TEXT_CHUNK_SIZE:
            result.append(para)
        else:
            sentences = _SENTENCE_RE.split(para)
            result.extend(s for s in sentences if s.strip())

    return result


def _split_paragraphs(text: str) -> list[str]:
    """빈 줄(\n\n) 기준으로 단락 분할합니다."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]

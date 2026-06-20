"""
app/core/rag/chunker.py 유닛 테스트.

외부 의존성 없는 순수 함수 테스트입니다.
"""

import pytest

from app.core.rag.chunker import (
    CODE_CHUNK_SIZE,
    CODE_OVERLAP,
    TEXT_CHUNK_SIZE,
    TEXT_OVERLAP,
    Chunk,
    chunk_document,
)


def test_empty_text_returns_empty():
    assert chunk_document("") == []
    assert chunk_document("   ") == []


def test_short_text_single_chunk():
    text = "짧은 텍스트"
    chunks = chunk_document(text)
    assert len(chunks) == 1
    assert chunks[0].content == text


def test_long_text_produces_multiple_chunks():
    text = "A" * (TEXT_CHUNK_SIZE + TEXT_OVERLAP + 1)
    chunks = chunk_document(text, doc_type="text")
    assert len(chunks) >= 2


def test_chunk_metadata_fields():
    text = "Hello world"
    chunks = chunk_document(text)
    meta = chunks[0].chunk_metadata
    assert "chunk_index" in meta
    assert "char_start" in meta
    assert "char_end" in meta
    assert meta["chunk_index"] == 0
    assert meta["char_start"] == 0
    assert meta["char_end"] == len(text)


def test_overlap_between_consecutive_chunks():
    text = "X" * (TEXT_CHUNK_SIZE * 2)
    chunks = chunk_document(text)
    assert len(chunks) >= 2
    first_end = chunks[0].chunk_metadata["char_end"]
    second_start = chunks[1].chunk_metadata["char_start"]
    assert first_end - second_start == TEXT_OVERLAP


def test_all_content_covered():
    """모든 청크의 content를 이어 붙이면 원문을 포함해야 합니다 (오버랩 제외)."""
    text = "B" * (TEXT_CHUNK_SIZE * 3)
    chunks = chunk_document(text)
    last_chunk = chunks[-1]
    assert last_chunk.chunk_metadata["char_end"] == len(text)


def test_code_type_uses_code_chunk_size():
    text = "C" * (CODE_CHUNK_SIZE + CODE_OVERLAP + 1)
    chunks = chunk_document(text, doc_type="python")
    assert len(chunks) >= 2
    # 첫 청크 크기가 CODE_CHUNK_SIZE 이하여야 함
    assert len(chunks[0].content) <= CODE_CHUNK_SIZE


def test_all_code_doc_types_use_code_strategy():
    code_types = ["python", "javascript", "typescript", "java", "kotlin",
                  "go", "rust", "cpp", "c", "git_diff"]
    text = "D" * (CODE_CHUNK_SIZE + CODE_OVERLAP + 1)
    for doc_type in code_types:
        chunks = chunk_document(text, doc_type=doc_type)
        assert len(chunks[0].content) <= CODE_CHUNK_SIZE, f"failed for {doc_type}"


def test_unknown_doc_type_uses_text_strategy():
    text = "E" * (TEXT_CHUNK_SIZE + TEXT_OVERLAP + 1)
    chunks_unknown = chunk_document(text, doc_type="unknown")
    chunks_text = chunk_document(text, doc_type="text")
    assert len(chunks_unknown) == len(chunks_text)


def test_chunk_index_sequential():
    text = "F" * (TEXT_CHUNK_SIZE * 3)
    chunks = chunk_document(text)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_metadata["chunk_index"] == i


def test_single_char_text():
    chunks = chunk_document("X")
    assert len(chunks) == 1
    assert chunks[0].content == "X"

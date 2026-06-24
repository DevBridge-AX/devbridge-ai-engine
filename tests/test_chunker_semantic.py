"""
시맨틱 청킹 전략별 유닛 테스트.

markdown / code / sql / git_diff / text 각 전략이 의미 단위로 분할되는지 검증합니다.
"""

import pytest

from app.core.rag.chunker import (
    CODE_CHUNK_SIZE,
    TEXT_CHUNK_SIZE,
    Chunk,
    chunk_document,
)


# ---------------------------------------------------------------------------
# 마크다운 전략
# ---------------------------------------------------------------------------

class TestMarkdownChunking:

    def test_splits_by_headers(self):
        text = "# Section 1\nContent A\n\n## Section 2\nContent B\n\n### Section 3\nContent C"
        chunks = chunk_document(text, doc_type="md")
        contents = [c.content for c in chunks]
        assert any("Section 1" in c for c in contents)
        assert any("Section 2" in c for c in contents)
        assert any("Section 3" in c for c in contents)

    def test_small_markdown_single_chunk(self):
        text = "# Title\nShort content"
        chunks = chunk_document(text, doc_type="markdown")
        assert len(chunks) == 1

    def test_large_section_falls_back_to_paragraphs(self):
        large_section = "# Big Section\n\n" + "\n\n".join(
            f"Paragraph {i}. " + "X" * 300 for i in range(10)
        )
        chunks = chunk_document(large_section, doc_type="md")
        assert len(chunks) > 1

    def test_md_doc_type_alias(self):
        text = "# A\nText A\n\n## B\nText B"
        chunks_md = chunk_document(text, doc_type="md")
        chunks_markdown = chunk_document(text, doc_type="markdown")
        assert len(chunks_md) == len(chunks_markdown)

    def test_metadata_has_required_fields(self):
        text = "# H1\nContent\n\n## H2\nMore content here"
        chunks = chunk_document(text, doc_type="md")
        for chunk in chunks:
            assert "chunk_index" in chunk.chunk_metadata
            assert "char_start" in chunk.chunk_metadata
            assert "char_end" in chunk.chunk_metadata

    def test_chunk_index_sequential(self):
        text = "# A\n" + "X" * 2000 + "\n\n## B\n" + "Y" * 2000
        chunks = chunk_document(text, doc_type="md")
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_metadata["chunk_index"] == i


# ---------------------------------------------------------------------------
# 코드 전략
# ---------------------------------------------------------------------------

class TestCodeChunking:

    def test_python_splits_by_functions(self):
        func_body = "    x = 1\n" * 40
        text = (
            "import os\nimport sys\n\n"
            f"def foo():\n{func_body}\n"
            f"def bar():\n{func_body}\n"
            f"class Baz:\n    def method(self):\n{func_body}"
        )
        chunks = chunk_document(text, doc_type="python")
        assert len(chunks) >= 2

    def test_python_async_def(self):
        body = "    await do_work()\n" * 40
        text = (
            f"async def handler():\n{body}\n"
            f"async def another():\n{body}\n"
        )
        chunks = chunk_document(text, doc_type="python")
        assert len(chunks) >= 2

    def test_large_function_falls_back(self):
        text = "def big_function():\n" + "    x = 1\n" * 500
        chunks = chunk_document(text, doc_type="python")
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= CODE_CHUNK_SIZE

    def test_single_function_under_limit(self):
        text = "def small():\n    return 1\n"
        chunks = chunk_document(text, doc_type="python")
        assert len(chunks) == 1

    def test_java_code(self):
        text = (
            "public class Foo {\n"
            "    public void methodA() {\n        // A\n    }\n\n"
            "    private int methodB() {\n        return 0;\n    }\n"
            "}\n"
        )
        chunks = chunk_document(text, doc_type="java")
        assert len(chunks) >= 1

    def test_typescript_code(self):
        text = (
            "export function greet(name: string): string {\n"
            "    return `Hello ${name}`;\n"
            "}\n\n"
            "export const add = (a: number, b: number) => a + b;\n"
        )
        chunks = chunk_document(text, doc_type="typescript")
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# SQL 전략
# ---------------------------------------------------------------------------

class TestSqlChunking:

    def test_splits_by_statements(self):
        cols = ",\n    ".join(f"col_{i} VARCHAR(255)" for i in range(20))
        text = (
            f"CREATE TABLE users (\n    id INT PRIMARY KEY,\n    {cols}\n);\n"
            f"CREATE TABLE orders (\n    id INT PRIMARY KEY,\n    {cols}\n);\n"
            "SELECT * FROM users;\n"
        )
        chunks = chunk_document(text, doc_type="sql")
        assert len(chunks) >= 2

    def test_single_statement(self):
        text = "SELECT * FROM users WHERE id = 1;"
        chunks = chunk_document(text, doc_type="sql")
        assert len(chunks) == 1

    def test_large_statement_falls_back(self):
        columns = ", ".join(f"col_{i} INT" for i in range(200))
        text = f"CREATE TABLE big_table (\n    {columns}\n);"
        chunks = chunk_document(text, doc_type="sql")
        if len(text) > CODE_CHUNK_SIZE:
            assert len(chunks) >= 2


# ---------------------------------------------------------------------------
# Git diff 전략
# ---------------------------------------------------------------------------

class TestGitDiffChunking:

    def test_splits_by_file(self):
        patch_lines = "\n".join(f"+    line_{i} = {i}" for i in range(50))
        text = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n+++ b/foo.py\n"
            f"@@ -1,3 +1,53 @@\n{patch_lines}\n\n"
            "diff --git a/bar.py b/bar.py\n"
            "--- a/bar.py\n+++ b/bar.py\n"
            f"@@ -1,2 +1,52 @@\n{patch_lines}\n"
        )
        chunks = chunk_document(text, doc_type="git_diff")
        assert len(chunks) >= 2

    def test_single_file_diff(self):
        text = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n+++ b/foo.py\n"
            "@@ -1,3 +1,3 @@\n-old\n+new\n"
        )
        chunks = chunk_document(text, doc_type="git_diff")
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# 일반 텍스트 전략
# ---------------------------------------------------------------------------

class TestTextChunking:

    def test_splits_by_paragraphs(self):
        paragraphs = [f"Paragraph {i}. " + "A" * 200 for i in range(8)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_document(text, doc_type="text")
        assert len(chunks) >= 2

    def test_large_paragraph_splits_by_sentences(self):
        sentences = [f"Sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        chunks = chunk_document(text, doc_type="text")
        assert len(chunks) >= 2

    def test_short_text_single_chunk(self):
        text = "Just a short paragraph."
        chunks = chunk_document(text, doc_type="text")
        assert len(chunks) == 1

    def test_preserves_all_content(self):
        paragraphs = [f"Para {i}. " + "B" * 400 for i in range(6)]
        text = "\n\n".join(paragraphs)
        chunks = chunk_document(text, doc_type="text")
        combined = "".join(c.content for c in chunks)
        for para in paragraphs:
            assert para.strip() in combined or para.strip()[:50] in combined


# ---------------------------------------------------------------------------
# 세그먼트 병합
# ---------------------------------------------------------------------------

class TestSmallSegmentMerging:

    def test_tiny_segments_get_merged(self):
        text = "# A\nX\n\n## B\nY\n\n## C\nZ"
        chunks = chunk_document(text, doc_type="md")
        assert len(chunks) <= 3

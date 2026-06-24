"""
app/core/rag/tokenizer.py 유닛 테스트.
"""

from app.core.rag.tokenizer import tokenize_for_bm25


class TestTokenizeForBM25:

    def test_korean_text(self):
        tokens = tokenize_for_bm25("인증 흐름이 어떻게 되나요")
        assert "인증" in tokens
        assert "어떻게" in tokens
        assert "되나요" in tokens

    def test_english_lowercase(self):
        tokens = tokenize_for_bm25("FastAPI Authentication")
        assert "fastapi" in tokens
        assert "authentication" in tokens

    def test_mixed_korean_english(self):
        tokens = tokenize_for_bm25("FastAPI 인증 설정 방법")
        assert "fastapi" in tokens
        assert "인증" in tokens
        assert "설정" in tokens

    def test_korean_bigram_expansion(self):
        tokens = tokenize_for_bm25("데이터베이스")
        assert "데이터베이스" in tokens
        assert "데이" in tokens
        assert "이터" in tokens
        assert "터베" in tokens

    def test_short_korean_no_bigram(self):
        tokens = tokenize_for_bm25("인증")
        assert "인증" in tokens
        bigrams = [t for t in tokens if len(t) == 2 and t != "인증"]
        assert len(bigrams) == 0

    def test_three_char_korean_no_bigram(self):
        tokens = tokenize_for_bm25("데이터")
        assert "데이터" in tokens
        assert "데이" not in tokens

    def test_punctuation_removed(self):
        tokens = tokenize_for_bm25("hello, world! 안녕하세요.")
        assert "," not in tokens
        assert "!" not in tokens
        assert "." not in tokens
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty_text(self):
        assert tokenize_for_bm25("") == []

    def test_whitespace_only(self):
        assert tokenize_for_bm25("   \n\t  ") == []

    def test_code_tokens(self):
        tokens = tokenize_for_bm25("def main(): return user_id")
        assert "def" in tokens
        assert "main" in tokens
        assert "return" in tokens
        assert "user_id" in tokens

    def test_numbers(self):
        tokens = tokenize_for_bm25("version 3.11 port 8080")
        assert "version" in tokens
        assert "3" in tokens
        assert "11" in tokens
        assert "8080" in tokens

    def test_special_chars_only(self):
        assert tokenize_for_bm25("!@#$%^&*()") == []

    def test_mixed_korean_english_compound(self):
        tokens = tokenize_for_bm25("JWT인증토큰 검증 로직")
        assert "jwt" in tokens or "JWT" in tokens
        assert "검증" in tokens

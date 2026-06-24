"""
하이브리드 검색(BM25 + Vector) 유닛 테스트.

RRF 병합 로직과 similarity_score 호환성을 검증합니다.
retrieve() 자체는 외부 의존성(임베딩 API, Chroma, DB)이 있으므로
_rrf_merge()와 핵심 로직을 단위 테스트합니다.
"""

import pytest

from app.core.rag.bm25_index import BM25SearchResult
from app.core.rag.retriever import _rrf_merge, _RRF_K


class TestRRFMerge:

    def test_both_sources_boost(self):
        """양쪽 모두에 등장하는 아이템이 더 높은 RRF 점수를 받습니다."""
        vector = [
            {"chunk_id": 1, "score": 0.9},
            {"chunk_id": 2, "score": 0.7},
            {"chunk_id": 3, "score": 0.5},
        ]
        bm25 = [
            BM25SearchResult(chunk_id=1, bm25_rank=0),
            BM25SearchResult(chunk_id=4, bm25_rank=1),
            BM25SearchResult(chunk_id=2, bm25_rank=2),
        ]
        result = _rrf_merge(vector, bm25, top_k=5)

        ids = [r["chunk_id"] for r in result]
        assert ids[0] == 1  # 양쪽 모두 1위/1위
        assert 1 in ids
        assert 2 in ids
        assert 3 in ids
        assert 4 in ids

    def test_vector_score_preserved(self):
        """vector_score에 원본 cosine 유사도가 보존됩니다."""
        vector = [{"chunk_id": 1, "score": 0.85}]
        bm25 = [BM25SearchResult(chunk_id=1, bm25_rank=0)]

        result = _rrf_merge(vector, bm25, top_k=5)
        assert result[0]["vector_score"] == 0.85

    def test_bm25_only_hit_zero_vector_score(self):
        """BM25에서만 매칭된 아이템은 vector_score=0.0입니다."""
        vector = [{"chunk_id": 1, "score": 0.9}]
        bm25 = [
            BM25SearchResult(chunk_id=1, bm25_rank=0),
            BM25SearchResult(chunk_id=99, bm25_rank=1),
        ]

        result = _rrf_merge(vector, bm25, top_k=5)
        bm25_only = [r for r in result if r["chunk_id"] == 99]
        assert len(bm25_only) == 1
        assert bm25_only[0]["vector_score"] == 0.0

    def test_vector_only_hit(self):
        """벡터에서만 매칭된 아이템도 결과에 포함됩니다."""
        vector = [
            {"chunk_id": 1, "score": 0.9},
            {"chunk_id": 2, "score": 0.6},
        ]
        bm25 = [BM25SearchResult(chunk_id=1, bm25_rank=0)]

        result = _rrf_merge(vector, bm25, top_k=5)
        ids = [r["chunk_id"] for r in result]
        assert 2 in ids

    def test_top_k_respected(self):
        """결과 수가 top_k를 초과하지 않습니다."""
        vector = [{"chunk_id": i, "score": 0.9 - i * 0.1} for i in range(10)]
        bm25 = [BM25SearchResult(chunk_id=i + 10, bm25_rank=i) for i in range(10)]

        result = _rrf_merge(vector, bm25, top_k=3)
        assert len(result) == 3

    def test_empty_vector(self):
        """벡터 결과가 비어있을 때 BM25 결과만 반환됩니다."""
        bm25 = [
            BM25SearchResult(chunk_id=1, bm25_rank=0),
            BM25SearchResult(chunk_id=2, bm25_rank=1),
        ]

        result = _rrf_merge([], bm25, top_k=5)
        assert len(result) == 2
        assert all(r["vector_score"] == 0.0 for r in result)

    def test_empty_bm25(self):
        """BM25 결과가 비어있을 때 벡터 결과만 반환됩니다."""
        vector = [
            {"chunk_id": 1, "score": 0.9},
            {"chunk_id": 2, "score": 0.7},
        ]

        result = _rrf_merge(vector, [], top_k=5)
        assert len(result) == 2
        assert result[0]["vector_score"] == 0.9

    def test_both_empty(self):
        """양쪽 모두 비어있으면 빈 리스트를 반환합니다."""
        result = _rrf_merge([], [], top_k=5)
        assert result == []

    def test_rrf_score_formula(self):
        """RRF 점수가 1/(k+rank+1) 공식을 따릅니다."""
        vector = [{"chunk_id": 1, "score": 0.9}]
        bm25 = [BM25SearchResult(chunk_id=1, bm25_rank=0)]

        result = _rrf_merge(vector, bm25, top_k=5)
        expected_rrf = 1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 1)
        assert abs(result[0]["rrf_score"] - expected_rrf) < 1e-10

    def test_duplicate_chunk_ids_in_same_source(self):
        """같은 소스에서 중복 chunk_id는 마지막 값이 사용됩니다."""
        vector = [
            {"chunk_id": 1, "score": 0.9},
            {"chunk_id": 1, "score": 0.7},
        ]

        result = _rrf_merge(vector, [], top_k=5)
        assert len(result) == 1

    def test_ordering_by_rrf_score(self):
        """결과가 RRF 점수 내림차순으로 정렬됩니다."""
        vector = [
            {"chunk_id": 1, "score": 0.5},
            {"chunk_id": 2, "score": 0.9},
        ]
        bm25 = [
            BM25SearchResult(chunk_id=2, bm25_rank=0),
            BM25SearchResult(chunk_id=3, bm25_rank=1),
        ]

        result = _rrf_merge(vector, bm25, top_k=5)
        rrf_scores = [r["rrf_score"] for r in result]
        assert rrf_scores == sorted(rrf_scores, reverse=True)


class TestGrndingCompatibility:
    """BM25-only 히트의 similarity_score가 grounding에 안전한지 검증."""

    def test_bm25_only_below_grounding_threshold(self):
        """BM25-only 히트는 similarity_score=0.0으로 grounding threshold(0.35) 미달."""
        vector = []
        bm25 = [BM25SearchResult(chunk_id=1, bm25_rank=0)]

        result = _rrf_merge(vector, bm25, top_k=5)
        assert result[0]["vector_score"] == 0.0
        assert result[0]["vector_score"] < 0.35

    def test_mixed_results_best_score_from_vector(self):
        """혼합 결과에서 best_score는 벡터 유사도 중 최대값입니다."""
        vector = [{"chunk_id": 1, "score": 0.45}]
        bm25 = [
            BM25SearchResult(chunk_id=1, bm25_rank=0),
            BM25SearchResult(chunk_id=2, bm25_rank=1),
        ]

        result = _rrf_merge(vector, bm25, top_k=5)
        best_vector_score = max(r["vector_score"] for r in result)
        assert best_vector_score == 0.45
        assert best_vector_score >= 0.35

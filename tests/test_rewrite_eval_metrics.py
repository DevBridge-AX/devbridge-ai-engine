"""
scripts/eval/rewrite_metrics.py 유닛 테스트 (A5).

전부 합성 데이터로 검증하며 실 API 호출이나 DB/벡터스토어 접근이 없습니다.
"""

from scripts.eval.rewrite_metrics import (
    compute_model_metrics,
    is_answer_like,
    is_passthrough_correct,
    keyword_hit_rate,
)


class TestIsAnswerLike:
    def test_normal_rewrite_is_not_answer_like(self):
        assert is_answer_like("그거 타임아웃 몇 초야?", "결제 API 타임아웃 시간") is False

    def test_length_over_four_times_original_is_answer_like(self):
        original = "그거 뭐야?"  # 5자
        rewritten = "그것" * 15  # 30자 > 5*4
        assert is_answer_like(original, rewritten) is True

    def test_length_over_80_chars_is_answer_like(self):
        original = "짧은 질문"
        rewritten = "가" * 81
        assert is_answer_like(original, rewritten) is True

    def test_newline_is_answer_like(self):
        assert is_answer_like("질문", "첫 줄\n둘째 줄") is True

    def test_ends_with_ibnida_is_answer_like(self):
        assert is_answer_like("질문", "결제 API 타임아웃은 30초입니다.") is True

    def test_ends_with_seubnida_is_answer_like(self):
        assert is_answer_like("질문", "재시도는 2회까지 진행합니다.") is True

    def test_quotes_is_answer_like(self):
        assert is_answer_like("질문", '재작성된 "쿼리"') is True

    def test_prefix_output_colon_is_answer_like(self):
        assert is_answer_like("질문", "출력: 결제 API 타임아웃") is True

    def test_prefix_rewrite_colon_is_answer_like(self):
        assert is_answer_like("질문", "재작성: 결제 API 타임아웃") is True

    def test_empty_rewrite_is_not_answer_like(self):
        assert is_answer_like("질문", "") is False
        assert is_answer_like("질문", "   ") is False


class TestKeywordHitRate:
    def test_all_keywords_present(self):
        assert keyword_hit_rate("결제 API 타임아웃 시간", ["결제", "타임아웃"]) == 1.0

    def test_partial_keywords_present(self):
        assert keyword_hit_rate("결제 API 관련 질문", ["결제", "타임아웃"]) == 0.5

    def test_no_keywords_present(self):
        assert keyword_hit_rate("전혀 다른 문장", ["결제", "타임아웃"]) == 0.0

    def test_case_insensitive_match(self):
        assert keyword_hit_rate("Access Token 만료", ["access token"]) == 1.0

    def test_empty_expected_keywords_is_vacuous_pass(self):
        assert keyword_hit_rate("아무 문장", []) == 1.0


class TestIsPassthroughCorrect:
    def test_identical_after_strip(self):
        assert is_passthrough_correct("고마워!", "  고마워!  ") is True

    def test_different_text_is_incorrect(self):
        assert is_passthrough_correct("고마워!", "감사합니다") is False


def _record(
    *,
    question="질문",
    rewritten="재작성됨",
    error=None,
    expected_keywords=None,
    expect_passthrough=False,
    expected_doc=None,
    doc_hit=None,
    latency_ms=100.0,
    prompt_tokens=50,
    completion_tokens=10,
):
    return {
        "question": question,
        "rewritten": rewritten,
        "error": error,
        "expected_keywords": expected_keywords or [],
        "expect_passthrough": expect_passthrough,
        "expected_doc": expected_doc,
        "doc_hit": doc_hit,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


class TestComputeModelMetrics:
    def test_empty_records_returns_zero_totals_and_none_rates(self):
        metrics = compute_model_metrics([])

        assert metrics["total_cases"] == 0
        assert metrics["error_count"] == 0
        assert metrics["answer_like_rate"] is None
        assert metrics["keyword_hit_rate"] is None
        assert metrics["passthrough_accuracy"] is None
        assert metrics["doc_hit_rate"] is None
        assert metrics["latency_p50_ms"] is None
        assert metrics["mean_prompt_tokens"] is None

    def test_error_records_counted_but_excluded_from_quality_metrics(self):
        records = [
            _record(rewritten="정상 재작성", expected_keywords=["재작성"]),
            _record(rewritten=None, error="TimeoutError: 요청 시간 초과", latency_ms=None,
                     prompt_tokens=None, completion_tokens=None),
        ]

        metrics = compute_model_metrics(records)

        assert metrics["total_cases"] == 2
        assert metrics["error_count"] == 1
        # 에러 레코드는 answer_like/keyword 집계에서 제외되어 정상 1건만 반영됨
        assert metrics["keyword_hit_rate"] == 1.0
        assert metrics["answer_like_rate"] == 0.0

    def test_answer_like_rate_computed_over_scored_records(self):
        records = [
            _record(question="그거 타임아웃은 몇 초야?", rewritten="결제 API 타임아웃 시간"),
            _record(question="그거 타임아웃은 몇 초야?", rewritten="결제 API 타임아웃은 30초입니다."),
        ]

        metrics = compute_model_metrics(records)

        assert metrics["answer_like_rate"] == 0.5

    def test_passthrough_accuracy_only_over_expect_passthrough_records(self):
        records = [
            _record(question="고마워!", rewritten="고마워!", expect_passthrough=True),
            _record(question="감사!", rewritten="완전히 다른 텍스트", expect_passthrough=True),
            _record(question="일반 질문", rewritten="재작성된 질문", expect_passthrough=False),
        ]

        metrics = compute_model_metrics(records)

        # expect_passthrough=False인 레코드는 분모에서 제외되어 2건 중 1건 정답 = 0.5
        assert metrics["passthrough_accuracy"] == 0.5

    def test_doc_hit_rate_ignores_records_without_expected_doc_or_measurement(self):
        records = [
            _record(expected_doc="payment-api", doc_hit=True),
            _record(expected_doc="payment-api", doc_hit=False),
            _record(expected_doc=None, doc_hit=None),  # chitchat: 집계 제외
            _record(expected_doc="auth-jwt-policy", doc_hit=None),  # 미측정(--no-retrieval): 집계 제외
        ]

        metrics = compute_model_metrics(records)

        assert metrics["doc_hit_rate"] == 0.5

    def test_latency_and_token_means(self):
        records = [
            _record(latency_ms=100.0, prompt_tokens=100, completion_tokens=20),
            _record(latency_ms=200.0, prompt_tokens=200, completion_tokens=40),
        ]

        metrics = compute_model_metrics(records)

        assert metrics["latency_p50_ms"] == 150.0
        assert metrics["mean_prompt_tokens"] == 150.0
        assert metrics["mean_completion_tokens"] == 30.0

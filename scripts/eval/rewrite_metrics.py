"""
rewrite 모델 평가용 순수 휴리스틱/집계 함수 (A5).

이 모듈은 API 호출이나 DB/벡터스토어 접근을 하지 않는 순수 함수만 포함합니다.
`scripts/eval/rewrite_eval.py`가 실 `query_rewriter.rewrite()` 호출 결과를 이 모듈의
함수로 채점하고 집계합니다. `tests/test_rewrite_eval_metrics.py`가 합성 데이터로
단위 테스트합니다(실 API 불필요).
"""

import math

# 구현은 중복을 피하기 위해 app 모듈에 있으며 여기서는 재노출만 한다.
from app.core.llm.rewrite_guard import is_answer_like  # noqa: F401


def keyword_hit_rate(rewritten: str, expected_keywords: list[str]) -> float:
    """expected_keywords 중 rewritten에 (대소문자 무시) 포함된 비율을 반환합니다.

    expected_keywords가 비어 있으면 검증할 대상이 없다는 뜻이므로 1.0(만점)을 반환합니다
    (chitchat 케이스처럼 키워드가 없는 경우).
    """
    if not expected_keywords:
        return 1.0

    text = (rewritten or "").casefold()
    hits = sum(1 for kw in expected_keywords if kw.casefold() in text)
    return hits / len(expected_keywords)


def is_passthrough_correct(original: str, rewritten: str) -> bool:
    """expect_passthrough=True인 케이스에서 rewritten이 원문과 (strip 후) 동일한지 확인합니다."""
    return (rewritten or "").strip() == (original or "").strip()


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: list[float], pct: float) -> float:
    """values의 pct(0~100) 백분위수를 선형 보간으로 계산합니다 (scripts/metrics_report.py와 동일 방식)."""
    if not values:
        return 0.0

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (pct / 100)
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[int(rank)]

    lower_value = ordered[int(lower)] * (upper - rank)
    upper_value = ordered[int(upper)] * (rank - lower)
    return lower_value + upper_value


def compute_model_metrics(records: list[dict]) -> dict:
    """한 모델의 케이스별 결과 레코드 목록을 집계합니다.

    각 record는 다음 키를 가질 것으로 기대합니다 (scripts/eval/rewrite_eval.py가 생성):
    - question: str (원래 질문)
    - rewritten: str | None (재작성 결과, 에러 시 None)
    - error: str | None (에러 메시지, 정상이면 None)
    - expected_keywords: list[str]
    - expect_passthrough: bool
    - expected_doc: str | None
    - doc_hit: bool | None (top-3 검색 결과에 expected_doc 포함 여부, 미측정이면 None)
    - latency_ms: float | None
    - prompt_tokens: int | None
    - completion_tokens: int | None

    반환값의 각 비율/평균 지표는 계산할 데이터가 전혀 없으면 None입니다.
    """
    total = len(records)
    ok_records = [r for r in records if not r.get("error")]
    error_count = total - len(ok_records)

    scored = [r for r in ok_records if r.get("rewritten") is not None]

    answer_like_flags = [is_answer_like(r["question"], r["rewritten"]) for r in scored]
    answer_like_rate = _mean([1.0 if f else 0.0 for f in answer_like_flags])

    keyword_rates = [
        keyword_hit_rate(r["rewritten"], r.get("expected_keywords") or []) for r in scored
    ]
    keyword_hit_rate_avg = _mean(keyword_rates)

    passthrough_records = [r for r in scored if r.get("expect_passthrough")]
    passthrough_flags = [
        1.0 if is_passthrough_correct(r["question"], r["rewritten"]) else 0.0
        for r in passthrough_records
    ]
    passthrough_accuracy = _mean(passthrough_flags)

    doc_hit_records = [
        r for r in scored if r.get("expected_doc") is not None and r.get("doc_hit") is not None
    ]
    doc_hit_flags = [1.0 if r["doc_hit"] else 0.0 for r in doc_hit_records]
    doc_hit_rate = _mean(doc_hit_flags)

    latencies = [
        float(r["latency_ms"]) for r in ok_records if isinstance(r.get("latency_ms"), (int, float))
    ]
    prompt_tokens = [
        float(r["prompt_tokens"])
        for r in ok_records
        if isinstance(r.get("prompt_tokens"), (int, float))
    ]
    completion_tokens = [
        float(r["completion_tokens"])
        for r in ok_records
        if isinstance(r.get("completion_tokens"), (int, float))
    ]

    return {
        "total_cases": total,
        "error_count": error_count,
        "answer_like_rate": answer_like_rate,
        "keyword_hit_rate": keyword_hit_rate_avg,
        "passthrough_accuracy": passthrough_accuracy,
        "doc_hit_rate": doc_hit_rate,
        "latency_p50_ms": _percentile(latencies, 50) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 95) if latencies else None,
        "mean_prompt_tokens": _mean(prompt_tokens),
        "mean_completion_tokens": _mean(completion_tokens),
    }

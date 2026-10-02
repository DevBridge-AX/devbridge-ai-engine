"""
시맨틱 캐시 임계치 오프라인 평가 (A9 part 1).

목적: 시맨틱 캐시의 유사도 임계치를 정할 근거 데이터를 만듭니다. 질문 쌍 데이터셋
(scripts/eval/datasets/cache_pairs.jsonl: same_intent 20쌍 + different_intent 20쌍)의
q1/q2를 임베딩해 쌍별 코사인 유사도를 수집하고, 임계치별 hit_rate(같은 질문을 hit)와
false_hit_rate(다른 질문을 hit = 오답)를 스윕해 권고 임계치를 계산합니다. 이 스크립트
자체는 config의 캐시 임계치를 변경하지 않습니다 — 값 변경은 이 결과를 근거로 한 별도
결정입니다.

이 모듈은 두 부분으로 나뉩니다.
- 순수 지표(scripts/eval/cache_eval_metrics.py): API/DB 호출이 없어
  tests/test_cache_eval_metrics.py가 합성 레코드로 단위 테스트합니다.
- 라이브 수집(CLI --collect): 실 임베딩 API를 호출합니다(임베딩 비용만 발생, LLM 호출 없음).

사용법:
    # 1) 라이브 수집 (실 임베딩 API 호출) — q1+q2 80건을 한 번에 임베딩하고
    #    쌍별 코사인 유사도를 캐시 jsonl로 저장합니다.
    RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect
    RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --limit 5   # 스모크용 소량 실행

    # 1-b) 라이브 수집 + LLM 재검증 — 임베딩 후 모든 쌍에 대해 verify_same_question(q1, q2)을
    #      순차 호출(REWRITE_MODEL, 쌍당 1회)해 후보 임계치를 오프라인으로 스윕할 수 있게 합니다.
    RUN_LIVE_LLM=1 python3 scripts/eval/cache_eval.py --collect --verify

    # 2) 오프라인 스윕 (API 호출 없음) — 캐시된 유사도로 임계치 0.85~0.99를 스윕합니다.
    #    verify 필드가 있는 캐시면 "임계치 + LLM 재검증" 스윕 섹션이 추가됩니다.
    python3 scripts/eval/cache_eval.py --from-cache data/eval/cache-20260930-120000.jsonl
    python3 scripts/eval/cache_eval.py --from-cache <file> --direct-threshold 0.95

캐시 레코드 스키마(jsonl 1줄 = 질문 쌍 1건):
    {id, kind, q1, q2, similarity, embedding_model}
    --verify 수집 시 추가: {verify_outcome, verify_same, verify_latency_ms,
    verify_prompt_tokens, verify_completion_tokens, verify_model, verify_prompt_version}
    (verify 필드가 없는 구버전 캐시도 그대로 읽습니다.)

비용: 임베딩 호출만 발생합니다(기본 40쌍 = 텍스트 80건, 배치 1~2회). --verify를 주면
쌍당 경량 LLM(REWRITE_MODEL) 호출이 1회씩 추가됩니다(기본 40회, max_tokens 16).
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

from scripts.eval.cache_eval_metrics import (
    DEFAULT_DIRECT_THRESHOLD,
    cosine_similarity,
    has_verify_fields,
    load_cache,
    load_dataset,
    recommend_threshold,
    render_markdown,
    render_verify_markdown,
    save_cache,
    sweep,
    verify_stats,
    verify_sweep,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "scripts" / "eval" / "datasets" / "cache_pairs.jsonl"
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "eval"


# ---------------------------------------------------------------------------
# 라이브 수집 (실 임베딩 API 호출) — CLI에서만 사용, 순수 함수와 분리
# ---------------------------------------------------------------------------

async def _collect_records(dataset_path: Path, limit: int | None, verify: bool = False) -> list[dict]:
    """모든 q1/q2를 한 번의 embed_texts 호출로 임베딩하고 쌍별 코사인 유사도를 계산합니다.

    verify=True면 임계치와 무관하게 모든 쌍에 verify_same_question(q1, q2)을 순차 호출해
    결과를 레코드에 함께 저장합니다(후보 임계치를 오프라인에서 스윕하기 위함).

    RUN_LIVE_LLM 게이트는 호출부(main)의 책임입니다.
    """
    from app.core.embeddings.embedder import embed_texts

    pairs = load_dataset(dataset_path)
    if limit is not None:
        pairs = pairs[:limit]
    if not pairs:
        return []

    texts: list[str] = []
    for pair in pairs:
        texts.append(pair["q1"])
        texts.append(pair["q2"])

    result = await embed_texts(texts)

    records = []
    for i, pair in enumerate(pairs):
        records.append(
            {
                "id": pair["id"],
                "kind": pair["kind"],
                "q1": pair["q1"],
                "q2": pair["q2"],
                "similarity": cosine_similarity(
                    result.embeddings[2 * i], result.embeddings[2 * i + 1]
                ),
                "embedding_model": result.embedding_model,
            }
        )
    if verify:
        from app.core.cache.verifier import CACHE_VERIFY_PROMPT_VERSION, verify_same_question
        from app.config import get_settings

        model = get_settings().rewrite_model
        print(f"재검증 모델(REWRITE_MODEL): {model}, 프롬프트 {CACHE_VERIFY_PROMPT_VERSION}")
        for record in records:
            result = await verify_same_question(record["q1"], record["q2"])
            record["verify_outcome"] = result.outcome
            record["verify_same"] = result.same
            record["verify_latency_ms"] = result.latency_ms
            record["verify_prompt_tokens"] = result.usage.prompt_tokens if result.usage else None
            record["verify_completion_tokens"] = result.usage.completion_tokens if result.usage else None
            record["verify_model"] = model
            record["verify_prompt_version"] = CACHE_VERIFY_PROMPT_VERSION
    return records


def _live_enabled() -> bool:
    """RUN_LIVE_LLM=1 AND settings.gms_api_key 비어있지 않음을 확인합니다(grounding_eval과 동일 게이트)."""
    if os.environ.get("RUN_LIVE_LLM") != "1":
        return False

    from app.config import get_settings

    return bool(get_settings().gms_api_key)


def _report(records: list[dict], direct_threshold: float = DEFAULT_DIRECT_THRESHOLD) -> str:
    rows = sweep(records)
    recommendations = {
        "false hit 0% 허용 (엄격)": recommend_threshold(rows, max_false_hit_rate=0.0),
        "false hit 2% 이하 허용": recommend_threshold(rows, max_false_hit_rate=0.02),
    }
    models = sorted({r["embedding_model"] for r in records if r.get("embedding_model")})
    report = render_markdown(records, rows, recommendations, embedding_model=", ".join(models) or "-")
    if has_verify_fields(records):
        report += render_verify_markdown(
            records,
            verify_sweep(records, direct_threshold=direct_threshold),
            verify_stats(records),
            direct_threshold=direct_threshold,
        )
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "시맨틱 캐시 임계치 오프라인 평가(A9). --collect로 질문 쌍 유사도를 수집해 "
            "캐시에 저장하거나, --from-cache로 저장된 캐시를 임계치 스윕합니다(API 호출 없음)."
        )
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="실 임베딩 API를 호출해 데이터셋 전 쌍의 코사인 유사도를 수집하고 결과를 data/eval/에 "
        "캐시로 저장합니다. RUN_LIVE_LLM=1 환경변수와 비어있지 않은 GMS_API_KEY가 필요합니다"
        "(임베딩 비용만 발생).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="--collect와 함께 사용: 모든 쌍에 LLM 재검증(REWRITE_MODEL)을 순차 호출해 verify_* 필드를 "
        "저장합니다(쌍당 LLM 호출 1회 비용 추가).",
    )
    parser.add_argument(
        "--direct-threshold",
        type=float,
        default=DEFAULT_DIRECT_THRESHOLD,
        help="--from-cache verify 섹션에서 검증 없이 즉시 hit로 보는 유사도 임계치 "
        f"(기본 {DEFAULT_DIRECT_THRESHOLD}, 운영 SEMANTIC_CACHE_THRESHOLD와 맞출 것)",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="--collect로 생성된 캐시 jsonl 경로. API 호출 없이 임계치 0.85~0.99를 스윕해 markdown을 출력합니다.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"질문 쌍 jsonl 경로 (기본: {_DEFAULT_DATASET.relative_to(_REPO_ROOT)})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="--collect 수집 시 처리할 쌍 수를 제한합니다(스모크 실행용, 비용 절감).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="--collect 결과 캐시 저장 경로 (기본: data/eval/cache-{YYYYMMDD-HHMMSS}.jsonl)",
    )
    args = parser.parse_args(argv)

    if args.collect and args.from_cache:
        parser.error("--collect와 --from-cache는 동시에 지정할 수 없습니다.")

    if args.verify and not args.collect:
        parser.error("--verify는 --collect와 함께만 사용할 수 있습니다.")

    if args.from_cache:
        print(_report(load_cache(args.from_cache), direct_threshold=args.direct_threshold))
        return 0

    if args.collect:
        if not _live_enabled():
            print(
                "RUN_LIVE_LLM=1 과 비어있지 않은 GMS_API_KEY가 필요합니다 "
                "(docs/semantic-cache-eval.md 참고). 임베딩 비용이 발생하는 실 API 호출입니다.",
                file=sys.stderr,
            )
            return 1

        records = asyncio.run(_collect_records(args.dataset, args.limit, args.verify))

        out_path = args.out
        if out_path is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            out_path = _DEFAULT_CACHE_DIR / f"cache-{timestamp}.jsonl"
        save_cache(out_path, records)
        print(f"수집 완료: {len(records)}건 -> {out_path}")
        return 0

    parser.error("--collect 또는 --from-cache 중 하나를 지정하세요.")
    return 2  # pragma: no cover — argparse.error()가 SystemExit을 던지므로 도달하지 않음


if __name__ == "__main__":
    raise SystemExit(main())

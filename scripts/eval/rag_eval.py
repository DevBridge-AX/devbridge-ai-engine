"""
RAG 검색 품질 + 접근 제어 평가 CLI (X2).

- --live: manifest의 가상 문서(기본 40건)를 tmp 워크스페이스에 실 임베딩으로 시딩하고, 케이스마다
  retriever.retrieve(access=...)를 호출해 결과를 data/eval/rag-{YYYYmmdd-HHMMSS}.jsonl로 저장합니다.
  임베딩 API만 호출합니다(문서 청크 + 질의 80건, LLM 호출 없음). RUN_LIVE_LLM=1 필요.
  --manifest / --cases 로 다른 문서 셋·케이스 파일을 지정할 수 있습니다.
- --rerank: --live와 함께 쓰며 settings.rerank_enabled(RERANK_ENABLED=true)가 켜져 있어야 합니다.
  retrieve(top_k=max(top_k, rerank_candidates)) 후 reranker.rerank를 호출해 레코드에 retrieved(후보 전체,
  원래 순서)와 reranked(상위 top_k), rerank_stage, rerank_latency_ms를 함께 저장합니다. 케이스당 LLM 호출이 1회 추가됩니다.
- --from-cache <path>: 저장된 jsonl로 지표를 집계해 markdown 표를 출력합니다(API 호출 없음).
  레코드에 reranked가 있으면 기준선(retrieved)·리랭크(reranked) 표와 카테고리별 차이 표를 함께 출력합니다.

사용법:
    RUN_LIVE_LLM=1 PYTHONPATH=. python3 scripts/eval/rag_eval.py --live [--limit N] [--top-k 5]
        [--manifest PATH] [--cases PATH] [--rerank] [--rerank-interval SEC] [--rerank-retries N]
    PYTHONPATH=. python3 scripts/eval/rag_eval.py --from-cache data/eval/rag-....jsonl

자세한 내용은 docs/rag-eval.md 를 참고하세요.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from scripts.eval.rag_docset import (
    DEFAULT_CASES,
    DEFAULT_MANIFEST,
    access_filter_for,
    load_cases,
    seed_docset_async,
)
from scripts.eval.rag_eval_metrics import (
    compute_metrics,
    has_reranked,
    render_diff_markdown,
    render_markdown,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE_DIR = _REPO_ROOT / "data" / "eval"


def load_cache(path: Path) -> list[dict]:
    """--live로 생성된 jsonl을 읽습니다. 파싱 실패한 줄은 건너뜁니다."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def save_cache(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _live_enabled() -> bool:
    """RUN_LIVE_LLM=1 AND settings.gms_api_key 비어있지 않음(grounding_eval과 동일 게이트)."""
    import os

    if os.environ.get("RUN_LIVE_LLM") != "1":
        return False
    from app.config import get_settings

    return bool(get_settings().gms_api_key)


async def _collect_records(
    limit: int | None,
    top_k: int,
    manifest_path: Path = DEFAULT_MANIFEST,
    cases_path: Path = DEFAULT_CASES,
    rerank: bool = False,
    rerank_interval: float = 0.0,
    rerank_retries: int = 0,
) -> list[dict]:
    """tmp 워크스페이스를 시드하고 케이스마다 retrieve를 호출합니다. 게이트는 main의 책임입니다.

    rerank=True면 후보 max(top_k, rerank_candidates)건을 가져와 리랭크하고 reranked 필드를 추가합니다.
    rerank_interval은 케이스 간 대기 초(리랭크 모델의 분당 요청 제한 회피용). rerank_retries는
    리랭크가 fallback:http_429로 끝났을 때 15초·30초·60초... 백오프 후 다시 시도하는 최대 횟수이며,
    재시도해도 실패하면 마지막 결과를 기록합니다(앱 경로는 재시도 없이 fail-open).
    """
    import os

    from app.config import get_settings
    from app.core.rag import reranker, retriever
    from app.db.vector_store import get_vector_store

    cases = load_cases(cases_path)
    if limit is not None:
        cases = cases[:limit]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        vector_store_path = tmp_path / "vector_store"
        metrics_dir = tmp_path / "metrics"
        vector_store_path.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        os.environ["VECTOR_STORE_PATH"] = str(vector_store_path)
        os.environ["METRICS_DIR"] = str(metrics_dir)
        get_settings.cache_clear()
        get_vector_store.cache_clear()

        db, workspace_id, doc_ids = await seed_docset_async(tmp_path, manifest_path)
        id_to_key = {str(v): k for k, v in doc_ids.items()}
        try:
            records = []
            retrieve_k = max(top_k, get_settings().rerank_candidates) if rerank else top_k

            def to_items(case, chunks) -> list[dict]:
                return [
                    {
                        "doc_key": id_to_key.get(str(c.source_id), f"unknown:{c.source_id}"),
                        "similarity": c.similarity_score,
                        "rank": rank,
                        # 청크 원문은 저장하지 않고 정답 문구 포함 여부(bool)만 기록합니다.
                        "substring_hit": any(s in c.content for s in case.expected_substrings),
                    }
                    for rank, c in enumerate(chunks, start=1)
                ]

            for case in cases:
                chunks = await retriever.retrieve(
                    case.question, workspace_id, db, retrieve_k, access=access_filter_for(case)
                )
                retrieved = to_items(case, chunks)
                rerank_fields: dict = {}
                if rerank:
                    if rerank_interval > 0 and records:
                        await asyncio.sleep(rerank_interval)
                    rr = await reranker.rerank(case.question, chunks, top_k=top_k)
                    for attempt in range(rerank_retries):
                        if rr.stage != "fallback:http_429":
                            break
                        await asyncio.sleep(15 * (2**attempt))
                        rr = await reranker.rerank(case.question, chunks, top_k=top_k)
                    rerank_fields = {
                        "reranked": to_items(case, rr.chunks),
                        "rerank_stage": rr.stage,
                        "rerank_latency_ms": rr.latency_ms,
                    }
                records.append(
                    {
                        "id": case.id,
                        "category": case.category,
                        "expected_doc_keys": case.expected_doc_keys,
                        "forbidden_doc_keys": case.forbidden_doc_keys,
                        "expected_substrings": case.expected_substrings,
                        "retrieved": retrieved,
                        # 1위와 2위 청크 유사도 차이(결과가 2건 미만이면 None)
                        "margin": (
                            retrieved[0]["similarity"] - retrieved[1]["similarity"]
                            if len(retrieved) >= 2
                            else None
                        ),
                        "top_similarity": max((r["similarity"] for r in retrieved), default=0.0),
                        # 시딩 조건 기록(CHUNK_CONTEXT_ENABLED/CHUNK_CONTEXT_BM25): 파일만 봐도 조건을 알 수 있게 한다.
                        "chunk_context": bool(get_settings().chunk_context_enabled),
                        "chunk_context_bm25": bool(get_settings().chunk_context_bm25),
                        **rerank_fields,
                    }
                )
            return records
        finally:
            db.close()
            get_settings.cache_clear()
            get_vector_store.cache_clear()


def _render_report(records: list[dict], k: int) -> str:
    """reranked가 있으면 기준선·리랭크·차이 표를, 없으면 기존 단일 표를 반환합니다."""
    if not has_reranked(records):
        return render_markdown(compute_metrics(records, k=k))
    # 기준선은 retrieved 상위 k건만 사용해 리랭크 결과(top_k건)와 같은 조건으로 비교한다.
    truncated = [
        {**r, "retrieved": [i for i in r["retrieved"] if i["rank"] <= k]} for r in records
    ]
    baseline = compute_metrics(truncated, k=k)
    reranked = compute_metrics(records, k=k, source="reranked")
    return "\n".join(
        [
            "기준선 (retrieved)",
            render_markdown(baseline),
            "",
            "리랭크 (reranked)",
            render_markdown(reranked),
            "",
            "차이 (리랭크 - 기준선)",
            render_diff_markdown(baseline, reranked),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "RAG 검색 품질/접근 제어 평가(X2). --live로 실 임베딩 API를 호출해 결과를 저장하거나, "
            "--from-cache로 저장된 결과를 집계합니다(API 호출 없음)."
        )
    )
    parser.add_argument("--live", action="store_true", help="RUN_LIVE_LLM=1 필요(임베딩 비용 발생).")
    parser.add_argument("--from-cache", type=Path, default=None, help="--live로 생성된 jsonl 경로.")
    parser.add_argument("--limit", type=int, default=None, help="--live 시 처리할 케이스 수 제한.")
    parser.add_argument("--top-k", type=int, default=5, help="검색 top_k 및 recall@k의 k (기본 5).")
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST, help="--live 시 사용할 문서 manifest.json 경로."
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="--live 시 사용할 케이스 jsonl 경로.")
    parser.add_argument(
        "--out", type=Path, default=None, help="--live 결과 저장 경로 (기본: data/eval/rag-{ts}.jsonl)"
    )
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="--live 시 리랭크 결과(reranked)도 수집. RERANK_ENABLED=true 필요(LLM 호출 발생).",
    )
    parser.add_argument(
        "--rerank-interval",
        type=float,
        default=0.0,
        help="--rerank 시 케이스 간 대기 초 (리랭크 모델의 분당 요청 제한 회피, 기본 0).",
    )
    parser.add_argument(
        "--rerank-retries",
        type=int,
        default=0,
        help="--rerank 시 429(요청 제한)로 fallback한 케이스를 백오프 후 다시 시도하는 최대 횟수 (기본 0).",
    )
    args = parser.parse_args(argv)

    if args.live and args.from_cache:
        parser.error("--live와 --from-cache는 동시에 지정할 수 없습니다.")
    if args.rerank and not args.live:
        parser.error("--rerank는 --live와 함께만 사용할 수 있습니다.")

    if args.from_cache:
        print(_render_report(load_cache(args.from_cache), args.top_k))
        return 0

    if args.live:
        if args.rerank:
            from app.config import get_settings

            if not get_settings().rerank_enabled:
                print(
                    "--rerank는 RERANK_ENABLED=true 설정이 필요합니다(settings.rerank_enabled가 꺼져 있음).",
                    file=sys.stderr,
                )
                return 1
        if not _live_enabled():
            print(
                "RUN_LIVE_LLM=1 과 비어있지 않은 GMS_API_KEY가 필요합니다 (docs/rag-eval.md 참고). "
                "임베딩 API를 실제 호출합니다.",
                file=sys.stderr,
            )
            return 1
        records = asyncio.run(
            _collect_records(
                args.limit,
                args.top_k,
                args.manifest,
                args.cases,
                rerank=args.rerank,
                rerank_interval=args.rerank_interval,
                rerank_retries=args.rerank_retries,
            )
        )
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = args.out or _DEFAULT_CACHE_DIR / f"rag-{stamp}.jsonl"
        save_cache(out_path, records)
        print(f"수집 완료: {len(records)}건 -> {out_path}\n")
        print(_render_report(records, args.top_k))
        return 0

    parser.error("--live 또는 --from-cache 중 하나를 지정하세요.")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

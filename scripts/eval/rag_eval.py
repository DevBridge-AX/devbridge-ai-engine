"""
RAG 검색 품질 + 접근 제어 평가 CLI (X2).

- --live: 가상 문서 10건을 tmp 워크스페이스에 실 임베딩으로 시딩하고, 케이스마다
  retriever.retrieve(access=...)를 호출해 결과를 data/eval/rag-{YYYYmmdd-HHMMSS}.jsonl로 저장합니다.
  임베딩 API만 호출합니다(문서 10건 + 질의 30건, LLM 호출 없음). RUN_LIVE_LLM=1 필요.
- --from-cache <path>: 저장된 jsonl로 지표를 집계해 markdown 표를 출력합니다(API 호출 없음).

사용법:
    RUN_LIVE_LLM=1 python3 scripts/eval/rag_eval.py --live [--limit N] [--top-k 5]
    python3 scripts/eval/rag_eval.py --from-cache data/eval/rag-....jsonl

자세한 내용은 docs/rag-eval.md 를 참고하세요.
"""

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from scripts.eval.rag_docset import DEFAULT_CASES, access_filter_for, load_cases, seed_docset_async
from scripts.eval.rag_eval_metrics import compute_metrics, render_markdown

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


async def _collect_records(limit: int | None, top_k: int) -> list[dict]:
    """tmp 워크스페이스를 시드하고 케이스마다 retrieve를 호출합니다. 게이트는 main의 책임입니다."""
    import os

    from app.config import get_settings
    from app.core.rag import retriever
    from app.db.vector_store import get_vector_store

    cases = load_cases(DEFAULT_CASES)
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

        db, workspace_id, doc_ids = await seed_docset_async(tmp_path)
        id_to_key = {str(v): k for k, v in doc_ids.items()}
        try:
            records = []
            for case in cases:
                chunks = await retriever.retrieve(
                    case.question, workspace_id, db, top_k, access=access_filter_for(case)
                )
                retrieved = [
                    {
                        "doc_key": id_to_key.get(str(c.source_id), f"unknown:{c.source_id}"),
                        "similarity": c.similarity_score,
                        "rank": rank,
                    }
                    for rank, c in enumerate(chunks, start=1)
                ]
                records.append(
                    {
                        "id": case.id,
                        "category": case.category,
                        "expected_doc_keys": case.expected_doc_keys,
                        "forbidden_doc_keys": case.forbidden_doc_keys,
                        "retrieved": retrieved,
                        "top_similarity": max((r["similarity"] for r in retrieved), default=0.0),
                    }
                )
            return records
        finally:
            db.close()
            get_settings.cache_clear()
            get_vector_store.cache_clear()


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
        "--out", type=Path, default=None, help="--live 결과 저장 경로 (기본: data/eval/rag-{ts}.jsonl)"
    )
    args = parser.parse_args(argv)

    if args.live and args.from_cache:
        parser.error("--live와 --from-cache는 동시에 지정할 수 없습니다.")

    if args.from_cache:
        print(render_markdown(compute_metrics(load_cache(args.from_cache), k=args.top_k)))
        return 0

    if args.live:
        if not _live_enabled():
            print(
                "RUN_LIVE_LLM=1 과 비어있지 않은 GMS_API_KEY가 필요합니다 (docs/rag-eval.md 참고). "
                "임베딩 API를 실제 호출합니다.",
                file=sys.stderr,
            )
            return 1
        records = asyncio.run(_collect_records(args.limit, args.top_k))
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_path = args.out or _DEFAULT_CACHE_DIR / f"rag-{stamp}.jsonl"
        save_cache(out_path, records)
        print(f"수집 완료: {len(records)}건 -> {out_path}\n")
        print(render_markdown(compute_metrics(records, k=args.top_k)))
        return 0

    parser.error("--live 또는 --from-cache 중 하나를 지정하세요.")
    return 2  # pragma: no cover


if __name__ == "__main__":
    sys.exit(main())

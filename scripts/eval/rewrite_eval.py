"""
REWRITE_MODEL 후보 비교 실행 스크립트 (A5).

scripts/eval/datasets/rewrite_cases.jsonl의 각 케이스에 대해 실제
app.core.llm.query_rewriter.rewrite()를 호출하고, scripts/eval/rewrite_metrics.py의
순수 함수로 채점/집계해 모델별 markdown 비교표를 출력합니다.

⚠️ 실 GMS API를 호출합니다(모델당 최대 25건 rewrite + --no-retrieval이 아니면
코퍼스 4개 문서 임베딩 1회). 비용에 주의하세요.

사용법 (반드시 레포 루트에서 `-m`으로 실행 — `scripts` 패키지가 editable install에
포함되어 있지 않아 `python3 scripts/eval/rewrite_eval.py` 직접 실행은
`ModuleNotFoundError: No module named 'scripts'`가 발생합니다):
    python3 -m scripts.eval.rewrite_eval --models claude-sonnet-4-6,claude-haiku-4-5-20251001,gemini-2.5-flash-lite
    python3 -m scripts.eval.rewrite_eval --models claude-sonnet-4-6 --limit 5 --no-retrieval
    python3 -m scripts.eval.rewrite_eval --from-cache data/eval/rewrite-20260930-120000.jsonl

--no-retrieval을 주지 않으면 시작 시 tests/live/fixtures/corpus 4개 문서를 tmp
워크스페이스에 실 임베딩으로 1회 인덱싱하고(scripts/eval/seed.py), 모델별로 재작성된
쿼리에 대해 retriever.retrieve(top_k=3)를 호출해 expected_doc이 top-3에 포함되는지
확인합니다(문서당 임베딩 호출은 시딩 1회뿐이며, 모델 수만큼 반복되지 않습니다).

OpenAI 계열(gpt-*) 후보를 --models에 추가할 경우 유의: provider.py의 OpenAI 분기는
call_rewrite의 max_tokens를 요청 바디에 포함하지 않습니다(OpenAI 계열 자체 정책).
"""

import argparse
import asyncio
import json
import os
import tempfile
import time
from datetime import datetime
from pathlib import Path

from scripts.eval.rewrite_metrics import compute_model_metrics

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO_ROOT / "scripts" / "eval" / "datasets" / "rewrite_cases.jsonl"
_DEFAULT_OUT_DIR = _REPO_ROOT / "data" / "eval"

# 1차 후보 3종 (A5 계획). gpt-5.4-nano는 로컬에서 현재 사용 중이나 위 OpenAI 주의사항
# 때문에 기본 후보에는 포함하지 않았다 — 필요하면 --models에 직접 추가한다.
_DEFAULT_MODELS = "claude-sonnet-4-6,claude-haiku-4-5-20251001,gemini-2.5-flash-lite"

_METRIC_COLUMNS = [
    ("total_cases", "케이스"),
    ("error_count", "에러"),
    ("answer_like_rate", "답변형 출력률"),
    ("keyword_hit_rate", "키워드 적중률"),
    ("passthrough_accuracy", "passthrough 정확도"),
    ("doc_hit_rate", "top-3 문서 적중률"),
    ("latency_p50_ms", "지연 p50(ms)"),
    ("latency_p95_ms", "지연 p95(ms)"),
    ("mean_prompt_tokens", "평균 prompt 토큰"),
    ("mean_completion_tokens", "평균 completion 토큰"),
]


def _load_dataset(path: Path, limit: int | None) -> list[dict]:
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        cases.append(json.loads(line))
    if limit is not None:
        cases = cases[:limit]
    return cases


def _seed_retrieval_workspace() -> tuple[object, str, str]:
    """tmp VECTOR_STORE_PATH/METRICS_DIR를 지정하고 코퍼스를 실 임베딩으로 인덱싱합니다.

    tests/live/conftest.py의 live_env/seeded_workspace와 동일한 패턴입니다.
    Returns:
        (db, workspace_id, tmp_dir)
    """
    tmp_dir = tempfile.mkdtemp(prefix="rewrite_eval_")
    vector_store_path = Path(tmp_dir) / "vector_store"
    metrics_dir = Path(tmp_dir) / "metrics"
    vector_store_path.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    os.environ["VECTOR_STORE_PATH"] = str(vector_store_path)
    os.environ["METRICS_DIR"] = str(metrics_dir)

    from app.config import get_settings
    from app.db.vector_store import get_vector_store

    get_settings.cache_clear()
    get_vector_store.cache_clear()

    from scripts.eval.seed import seed_workspace

    db, workspace_id = seed_workspace(Path(tmp_dir))
    return db, workspace_id, tmp_dir


async def _run_model(
    model: str,
    cases: list[dict],
    db,
    workspace_id: str | None,
    skip_retrieval: bool,
) -> list[dict]:
    """한 모델에 대해 데이터셋 전체를 rewrite() 호출하고 채점용 레코드를 만듭니다."""
    from app.core.llm import query_rewriter
    from app.core.rag import retriever

    records: list[dict] = []
    for case in cases:
        question = case["question"]
        history = case.get("history") or []

        start = time.perf_counter()
        rewritten: str | None = None
        prompt_tokens = completion_tokens = None
        error: str | None = None
        try:
            rewritten, usage = await query_rewriter.rewrite(question, history)
            if usage is not None:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
        except Exception as exc:  # noqa: BLE001 - 케이스별 에러를 기록하고 계속 진행
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = (time.perf_counter() - start) * 1000

        expected_doc = case.get("expected_doc")
        doc_hit: bool | None = None
        if (
            not skip_retrieval
            and error is None
            and expected_doc is not None
            and workspace_id is not None
        ):
            try:
                results = await retriever.retrieve(rewritten, workspace_id, db, top_k=3)
                doc_hit = any(chunk.title == expected_doc for chunk in results)
            except Exception:  # noqa: BLE001 - 검색 실패는 doc_hit 미측정으로만 처리
                doc_hit = None

        records.append(
            {
                "model": model,
                "id": case["id"],
                "category": case.get("category"),
                "question": question,
                "rewritten": rewritten,
                "error": error,
                "latency_ms": latency_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "expected_keywords": case.get("expected_keywords") or [],
                "expect_passthrough": bool(case.get("expect_passthrough")),
                "expected_doc": expected_doc,
                "doc_hit": doc_hit,
            }
        )

    return records


def _format_metric(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _print_table(all_records: list[dict]) -> None:
    models = sorted({r["model"] for r in all_records})
    metrics_by_model = {
        model: compute_model_metrics([r for r in all_records if r["model"] == model])
        for model in models
    }

    header = "| model | " + " | ".join(label for _, label in _METRIC_COLUMNS) + " |"
    separator = "| --- | " + " | ".join("---" for _ in _METRIC_COLUMNS) + " |"
    print("\n## Rewrite 모델 비교 결과\n")
    print(header)
    print(separator)
    for model in models:
        metrics = metrics_by_model[model]
        row = [model] + [_format_metric(metrics[key]) for key, _ in _METRIC_COLUMNS]
        print("| " + " | ".join(row) + " |")
    print()


def _save_records(all_records: list[dict]) -> Path:
    _DEFAULT_OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = _DEFAULT_OUT_DIR / f"rewrite-{timestamp}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out_path


def _load_cache(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="REWRITE_MODEL 후보 모델을 비교 평가합니다 (A5, docs/rewrite-model-eval.md 참고).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "기본 후보 3종: claude-sonnet-4-6, claude-haiku-4-5-20251001, "
            "gemini-2.5-flash-lite.\n"
            "참고: gpt-5.4-nano 등 OpenAI 계열을 추가하면 provider.py가 call_rewrite의 "
            "max_tokens를 요청 바디에 포함하지 않는다(OpenAI 분기 고유 동작)."
        ),
    )
    parser.add_argument(
        "--models",
        default=_DEFAULT_MODELS,
        help=f"쉼표로 구분된 REWRITE_MODEL 후보 목록 (기본값: {_DEFAULT_MODELS})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="데이터셋 앞에서부터 N개 케이스만 사용 (기본값: 전체 25건)",
    )
    parser.add_argument(
        "--from-cache",
        type=Path,
        default=None,
        help="실 API를 호출하지 않고 저장된 jsonl 캐시 파일로 오프라인 재집계",
    )
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="재작성 쿼리의 top-3 문서 적중 여부를 확인하지 않음 (워크스페이스 시딩/임베딩 호출 생략)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=_DEFAULT_DATASET,
        help=f"평가 데이터셋 경로 (기본값: {_DEFAULT_DATASET.relative_to(_REPO_ROOT)})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.from_cache is not None:
        all_records = _load_cache(args.from_cache)
        _print_table(all_records)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cases = _load_dataset(args.dataset, args.limit)

    db = workspace_id = None
    tmp_dir: str | None = None
    if not args.no_retrieval:
        db, workspace_id, tmp_dir = _seed_retrieval_workspace()

    all_records: list[dict] = []
    try:
        for model in models:
            os.environ["REWRITE_MODEL"] = model

            from app.config import get_settings

            get_settings.cache_clear()

            print(f"[rewrite_eval] {model} 실행 중 ({len(cases)}건)...")
            records = asyncio.run(_run_model(model, cases, db, workspace_id, args.no_retrieval))
            all_records.extend(records)
    finally:
        if db is not None:
            db.close()

    out_path = _save_records(all_records)
    print(f"결과 파일: {out_path.relative_to(_REPO_ROOT)}")
    if tmp_dir is not None:
        print(f"(시드 워크스페이스 tmp 디렉터리: {tmp_dir}, 자동 삭제되지 않으므로 필요 시 수동 정리)")

    _print_table(all_records)


if __name__ == "__main__":
    main()

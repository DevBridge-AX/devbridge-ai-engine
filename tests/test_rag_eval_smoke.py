"""
scripts/eval/rag_eval.py --live 경로 오프라인 스모크 테스트 (X2).

유료 L1 실행(실 임베딩: 문서 10건 + 질의 30건) 전에 배관(plumbing)이 깨지지 않았는지만 확인합니다.
임베딩은 결정적 가짜(문자 bigram 해시 벡터)로 대체하며 네트워크/LLM 호출이 없습니다.
검색 품질(recall 등)은 가짜 임베딩이라 의미가 없으므로 단언하지 않고, 구조/매핑/접근 제어만 검증합니다.
"""

import hashlib
import json
import math

import pytest

from app.config import get_settings
from app.core.embeddings.embedder import EmbedResult
from app.db.vector_store import get_vector_store
from scripts.eval import rag_eval
from scripts.eval.rag_docset import doc_allowed, load_cases, load_manifest

_DIM = 256
_TOP_K = 5
_REQUIRED_KEYS = {"id", "category", "expected_doc_keys", "forbidden_doc_keys", "retrieved", "top_similarity"}


def _fake_vector(text: str) -> list[float]:
    """문자 bigram을 해시해 _DIM 차원 버킷에 누적하고 L2 정규화합니다(어휘가 겹칠수록 코사인 유사도 상승)."""
    vec = [0.0] * _DIM
    compact = "".join(text.split())
    grams = [compact[i : i + 2] for i in range(len(compact) - 1)] or [compact or " "]
    for gram in grams:
        bucket = int(hashlib.md5(gram.encode("utf-8")).hexdigest(), 16) % _DIM
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


async def _fake_embed_texts(texts, **kwargs):
    return EmbedResult(
        embeddings=[_fake_vector(t) for t in texts],
        embedding_model="fake-embedding",
        embedding_model_version="fake-1",
        total_tokens=0.0,
    )


async def _exploding_embed_texts(texts, **kwargs):
    raise AssertionError("게이트가 닫혀 있으면 임베딩을 호출하면 안 됩니다.")


def _patch_embed(monkeypatch, fake) -> None:
    monkeypatch.setattr("app.pipelines.document_ingestion.embed_texts", fake)
    monkeypatch.setattr("app.core.rag.retriever.embed_texts", fake)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    """_collect_records가 os.environ을 직접 바꾸므로 setenv로 원복을 보장하고, 캐시를 전후로 비웁니다."""
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vs-placeholder"))
    monkeypatch.setenv("METRICS_DIR", str(tmp_path / "metrics-placeholder"))
    monkeypatch.delenv("RUN_LIVE_LLM", raising=False)
    get_settings.cache_clear()
    get_vector_store.cache_clear()
    yield
    get_settings.cache_clear()
    get_vector_store.cache_clear()


@pytest.fixture
def records(monkeypatch):
    import asyncio

    _patch_embed(monkeypatch, _fake_embed_texts)
    return asyncio.run(rag_eval._collect_records(None, _TOP_K))


def test_collect_records_structure(records):
    cases = load_cases()
    assert len(records) == len(cases) == 30
    assert [r["id"] for r in records] == [c.id for c in cases]
    for record in records:
        assert _REQUIRED_KEYS <= record.keys()
        retrieved = record["retrieved"]
        assert len(retrieved) <= _TOP_K
        assert [r["rank"] for r in retrieved] == list(range(1, len(retrieved) + 1))
        assert all(not r["doc_key"].startswith("unknown:") for r in retrieved)
        assert isinstance(record["top_similarity"], float)
    # 가짜 임베딩이라도 검색 자체는 동작해야 함(전부 빈 결과면 배관 문제)
    assert any(r["retrieved"] for r in records)


def test_collect_records_acl(records):
    manifest = {e.doc_key: e for e in load_manifest()}
    cases = {c.id: c for c in load_cases()}
    checked = 0
    for record in records:
        if record["category"] not in ("acl_task", "acl_restricted"):
            continue
        case = cases[record["id"]]
        for item in record["retrieved"]:
            assert item["doc_key"] not in record["forbidden_doc_keys"], record["id"]
            assert doc_allowed(manifest[item["doc_key"]], case.access), record["id"]
        checked += 1
    assert checked == 8


def test_env_restored_after_collect(records, tmp_path):
    import os

    # _collect_records가 바꾼 경로가 monkeypatch 값이 아니라 이미 정리된 tmp를 가리키더라도,
    # 캐시는 비워져 있어 다음 get_settings()가 현재 환경을 다시 읽어야 합니다.
    assert get_settings.cache_info().currsize == 0
    assert get_vector_store.cache_info().currsize == 0
    assert os.environ["VECTOR_STORE_PATH"]  # 값 자체는 fixture 종료 시 monkeypatch가 원복


def test_cache_roundtrip_and_report(records, tmp_path):
    path = tmp_path / "out" / "rag.jsonl"
    rag_eval.save_cache(path, records)
    loaded = rag_eval.load_cache(path)
    assert loaded == json.loads(json.dumps(records, ensure_ascii=False)) == records

    metrics = rag_eval.compute_metrics(loaded, k=_TOP_K)
    assert {"k", "overall", "categories"} <= metrics.keys()
    assert {"n", "n_scored", "recall_at_k", "mrr", "hit_at_1", "leak_count"} <= metrics["overall"].keys()
    assert metrics["overall"]["n"] == 30
    # leak은 ACL 카테고리만 집계(distractor의 forbidden 노출은 forbidden_exposed_count로 분리)
    assert metrics["overall"]["leak_count"] == 0
    assert "top_similarity_mean" in metrics["categories"]["out_of_corpus"]

    report = rag_eval.render_markdown(metrics)
    assert f"recall@{_TOP_K}" in report
    assert "| overall |" in report
    assert "out_of_corpus top similarity" in report


def test_main_live_end_to_end(monkeypatch, tmp_path, capsys):
    _patch_embed(monkeypatch, _fake_embed_texts)
    monkeypatch.setattr(rag_eval, "_live_enabled", lambda: True)
    out = tmp_path / "rag.jsonl"

    assert rag_eval.main(["--live", "--out", str(out)]) == 0

    lines = [line for line in out.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 30
    stdout = capsys.readouterr().out
    assert "수집 완료: 30건" in stdout
    assert "| 카테고리 |" in stdout


def test_main_live_gate_closed(monkeypatch, tmp_path, capsys):
    _patch_embed(monkeypatch, _exploding_embed_texts)
    monkeypatch.setattr(rag_eval, "_live_enabled", lambda: False)
    out = tmp_path / "rag.jsonl"

    assert rag_eval.main(["--live", "--out", str(out)]) == 1
    assert not out.exists()
    assert "RUN_LIVE_LLM=1" in capsys.readouterr().err

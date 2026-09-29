"""
chat_pipeline 응답 지표 계측(B-3) 유닛 테스트.

mock LLM/검색/그라운딩으로 정상 응답 경로와 조기 종료(비그라운딩) 경로를 각각 실행해
chat_metrics.jsonl에 1건씩 기록되는지, 필드가 모두 존재/비음수인지, 질문 원문·답변
텍스트·user_id가 포함되지 않는지 검증합니다. metrics_dir는 tmp_path로 돌립니다.
"""

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from app.core import chat_pipeline, metrics
from app.core.llm.provider import LLMUsage
from app.core.rag.grounding import GroundingResult
from app.core.rag.retriever import RetrievedChunk
from app.schemas.chat import ChatRequest


def _request(**overrides) -> ChatRequest:
    payload = {
        "session_id": "session-abc",
        "content": "배포 절차 알려줘",
        "conversation_history": [],
        "workspace_id": "ws-1",
        "user_id": "user-1",
        "role": "planner",
        **overrides,
    }
    return ChatRequest(**payload)


@pytest.fixture(autouse=True)
def metrics_tmp_dir(monkeypatch, tmp_path):
    """record_metric이 실제로 사용하는 app.core.metrics.get_settings를 tmp_path로 돌립니다."""
    monkeypatch.setattr(
        metrics,
        "get_settings",
        lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(tmp_path)),
    )
    return tmp_path


def _read_records(tmp_path) -> list[dict]:
    path = tmp_path / "chat_metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run(request: ChatRequest) -> list:
    async def consume():
        return [event async for event in chat_pipeline.run(request, db=None)]

    return asyncio.run(consume())


class TestNonGroundablePath:

    def test_records_one_metric_without_pii(self, monkeypatch, metrics_tmp_dir):
        async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None):
            return []

        async def fake_assess(chunks, query):
            return GroundingResult(is_groundable=False, confidence=0.0)

        monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
        monkeypatch.setattr(
            chat_pipeline, "get_settings", lambda: SimpleNamespace(main_model="fake-main")
        )

        request = _request(session_id="session-abc")
        events = _run(request)

        assert events[-1].event == "done"
        assert events[-1].data["is_groundable"] is False

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]

        expected_hash = hashlib.sha256(b"session-abc").hexdigest()[:12]
        assert record["session_id_hash"] == expected_hash
        assert record["is_groundable"] is False
        assert record["grounding_stage"] == "threshold"
        assert record["chunk_count"] == 0
        assert record["top_similarity"] == 0.0
        assert record["turn"] == 1
        assert record["ttft_ms"] is None
        assert record["total_ms"] >= 0
        assert record["retrieve_ms"] >= 0
        assert record["grounding_ms"] >= 0
        assert record["llm_ms"] is None

        raw = json.dumps(record, ensure_ascii=False)
        assert "session_id" not in record
        assert "question" not in record
        assert "user_id" not in record
        assert request.content not in raw
        assert request.user_id not in raw
        assert request.session_id not in raw


class TestGroundablePath:

    def test_records_one_metric_with_token_usage(self, monkeypatch, metrics_tmp_dir):
        chunk = RetrievedChunk(
            chunk_id=1,
            source_type="document",
            source_id="doc-1",
            title="제목",
            content="내용",
            similarity_score=0.8,
        )

        async def fake_retrieve(query, workspace_id, db, top_k=5, access=None, timer=None):
            return [chunk]

        async def fake_assess(chunks, query):
            return GroundingResult(
                is_groundable=True,
                confidence=0.9,
                llm_usage=LLMUsage(model="gemini-grounding", prompt_tokens=10, completion_tokens=5),
            )

        async def fake_stream(messages, system_prompt):
            yield "안녕", None
            yield "하세요", None
            yield None, LLMUsage(model="claude-main", prompt_tokens=100, completion_tokens=20)

        monkeypatch.setattr(chat_pipeline.retriever, "retrieve", fake_retrieve)
        monkeypatch.setattr(chat_pipeline.grounding, "assess", fake_assess)
        monkeypatch.setattr(chat_pipeline.llm, "call_main_stream", fake_stream)

        request = _request(session_id="session-xyz")
        events = _run(request)

        assert events[-1].event == "done"
        assert events[-1].data["is_groundable"] is True
        token_events = [e for e in events if e.event == "token"]
        assert len(token_events) == 2

        records = _read_records(metrics_tmp_dir)
        assert len(records) == 1
        record = records[0]

        assert record["is_groundable"] is True
        assert record["grounding_stage"] == "llm"
        assert record["chunk_count"] == 1
        assert record["top_similarity"] == pytest.approx(0.8)
        assert record["ttft_ms"] is not None and record["ttft_ms"] >= 0
        assert record["llm_ms"] >= 0
        assert record["total_ms"] >= 0
        assert record["main_model"] == "claude-main"
        assert record["main_prompt_tokens"] == 100
        assert record["main_completion_tokens"] == 20
        assert record["grounding_model"] == "gemini-grounding"
        assert record["grounding_prompt_tokens"] == 10

        raw = json.dumps(record, ensure_ascii=False)
        assert "안녕" not in raw
        assert "하세요" not in raw
        assert request.content not in raw
        assert request.user_id not in raw

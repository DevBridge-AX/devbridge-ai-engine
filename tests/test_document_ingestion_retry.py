"""
문서 인덱싱 재시도/재인덱싱 유닛 테스트.

- 같은 문서를 다시 인덱싱하면 기존 청크(MySQL)와 벡터가 새 실행 결과로 교체되는지
- store 단계 중간 실패 시 이번 실행에서 추가한 벡터만 보상 삭제되고 기존 청크/벡터는 유지되는지
- POST /ingestion/document/retry 상태 가드(COMPLETED/진행 중 409, FAILED 202, 미존재 404)
- 프로세스 단위 동시 실행 가드(ingest_document)
임베딩 API·Chroma·BM25는 mock, DB는 인메모리 SQLite를 사용합니다.
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api import ingestion as ingestion_api
from app.config import get_settings
from app.core import metrics
from app.db.models import ChunkSourceType, DocumentChunk, KnowledgeDocument
from app.db.session import get_db
from app.main import app
from app.pipelines import document_ingestion

WS = "ws-1"
DOC = "doc-1"
LONG_DOC = "\n\n".join(f"## 소제목 {i}\n\n{'본문 문장입니다. ' * 60}" for i in range(4))


class FakeVectorStore:
    """add/delete 호출을 기록하는 인메모리 벡터 스토어. fail_on_add_call번째 add에서 예외."""

    def __init__(self) -> None:
        self.vectors: set[str] = set()
        self.deleted: list[str] = []
        self.add_calls = 0
        self.fail_on_add_call: int | None = None

    def add(self, **kw) -> None:
        self.add_calls += 1
        if self.fail_on_add_call is not None and self.add_calls == self.fail_on_add_call:
            raise RuntimeError("vector add boom")
        self.vectors.add(kw["vector_id"])

    def delete(self, vector_id: str, workspace_id: str) -> None:
        self.deleted.append(vector_id)
        self.vectors.discard(vector_id)


@pytest.fixture
def engine():
    eng = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    DocumentChunk.__table__.create(eng)
    KnowledgeDocument.__table__.create(eng)
    return eng


@pytest.fixture
def db(engine):
    with Session(engine) as session:
        yield session


@pytest.fixture
def store(monkeypatch, tmp_path):
    fake = FakeVectorStore()

    async def fake_embed(texts):
        return SimpleNamespace(
            embeddings=[[0.1, 0.2] for _ in texts],
            embedding_model="fake",
            embedding_model_version="v0",
            total_tokens=len(texts),
        )

    monkeypatch.setattr(document_ingestion, "embed_texts", fake_embed)
    monkeypatch.setattr(document_ingestion, "get_vector_store", lambda: fake)
    monkeypatch.setattr(
        document_ingestion, "get_bm25_manager", lambda: SimpleNamespace(invalidate=lambda w: None)
    )
    monkeypatch.setattr(
        document_ingestion,
        "get_semantic_cache",
        lambda: SimpleNamespace(invalidate_workspace=lambda w: None),
    )
    monkeypatch.setattr(document_ingestion, "log_embedding_usage", lambda *a, **kw: None)
    monkeypatch.setattr(
        metrics, "get_settings", lambda: SimpleNamespace(metrics_enabled=True, metrics_dir=str(tmp_path / "metrics"))
    )
    document_ingestion._IN_FLIGHT.clear()
    yield fake
    document_ingestion._IN_FLIGHT.clear()


@pytest.fixture
def doc_file(tmp_path):
    path = tmp_path / "doc.md"
    path.write_text(LONG_DOC, encoding="utf-8")
    return str(path)


def _chunks(db) -> list[DocumentChunk]:
    db.expire_all()
    return list(db.execute(select(DocumentChunk)).scalars().all())


def _add_doc(db, status: str, doc_id: str = DOC, file_path: str = "/x") -> None:
    now = datetime.now(timezone.utc)
    db.add(
        KnowledgeDocument(
            id=doc_id, workspace_id=WS, source_id="src", title="t", document_type="markdown",
            file_path=file_path, analysis_status=status, created_at=now, updated_at=now,
        )
    )
    db.commit()


async def _ingest(doc_file):
    await document_ingestion.ingest_document(WS, DOC, doc_file, "markdown")


@pytest.fixture
def session_local(engine, monkeypatch):
    sl = sessionmaker(bind=engine)
    monkeypatch.setattr(document_ingestion, "SessionLocal", sl)
    return sl


class TestReplaceOnReingest:

    def test_second_run_replaces_first_run_chunks_and_vectors(
        self, db, store, doc_file, session_local
    ):
        _add_doc(db, "PENDING", file_path=doc_file)
        asyncio.run(_ingest(doc_file))
        first = _chunks(db)
        first_ids = {c.vector_id for c in first}
        assert len(first) > 1
        assert store.vectors == first_ids

        asyncio.run(_ingest(doc_file))
        second = _chunks(db)
        second_ids = {c.vector_id for c in second}

        assert len(second) == len(first)  # 중복 누적 없음
        assert second_ids.isdisjoint(first_ids)
        assert store.vectors == second_ids
        assert set(store.deleted) == first_ids  # 커밋 후 기존 벡터 삭제

    def test_other_documents_chunks_are_untouched(self, db, store, doc_file, session_local):
        db.add(
            DocumentChunk(
                workspace_id=WS, source_type=ChunkSourceType.DOCUMENT, source_id="other-doc",
                content="x", embedding_model="fake", embedding_model_version="v0",
                vector_id="other-vec",
            )
        )
        db.commit()
        _add_doc(db, "PENDING", file_path=doc_file)

        asyncio.run(_ingest(doc_file))

        assert "other-vec" in {c.vector_id for c in _chunks(db)}
        assert "other-vec" not in store.deleted

    def test_old_vector_delete_error_keeps_completed(self, db, store, doc_file, session_local):
        _add_doc(db, "PENDING", file_path=doc_file)
        asyncio.run(_ingest(doc_file))

        def boom(vector_id, workspace_id):
            raise RuntimeError("delete boom")

        store.delete = boom
        asyncio.run(_ingest(doc_file))  # 예외 없음

        status = db.execute(select(KnowledgeDocument.analysis_status)).scalar_one()
        assert status == "COMPLETED"

    def test_commit_failure_keeps_old_vectors(self, db, engine, store, doc_file, monkeypatch):
        _add_doc(db, "PENDING", file_path=doc_file)
        monkeypatch.setattr(document_ingestion, "SessionLocal", sessionmaker(bind=engine))
        asyncio.run(_ingest(doc_file))
        old_ids = {c.vector_id for c in _chunks(db)}

        class FailingCommitSession(Session):
            failed = False

            def commit(self):
                # 첫 commit(COMPLETED 갱신)만 실패시키고, 이후(FAILED 갱신)는 정상 동작.
                if not FailingCommitSession.failed:
                    FailingCommitSession.failed = True
                    raise RuntimeError("commit boom")
                super().commit()

        monkeypatch.setattr(
            document_ingestion, "SessionLocal", sessionmaker(bind=engine, class_=FailingCommitSession)
        )
        store.deleted.clear()
        asyncio.run(_ingest(doc_file))

        assert {c.vector_id for c in _chunks(db)} == old_ids  # rollback으로 기존 행 유지
        assert not set(store.deleted) & old_ids  # 기존 벡터 삭제 안 함
        assert old_ids <= store.vectors
        status = db.execute(select(KnowledgeDocument.analysis_status)).scalar_one()
        assert status == "FAILED"


class TestStoreFailureCompensation:

    def test_failed_store_removes_new_vectors_and_keeps_old(
        self, db, engine, store, doc_file, monkeypatch
    ):
        # 1차 정상 인덱싱
        asyncio.run(document_ingestion._run(db, WS, DOC, doc_file, "markdown"))
        db.commit()
        old_ids = {c.vector_id for c in _chunks(db)}
        assert len(old_ids) > 2

        # 2차: 3번째 add에서 실패 (앞의 2개는 이미 추가됨)
        _add_doc(db, "FAILED", file_path=doc_file)
        store.fail_on_add_call = store.add_calls + 3
        monkeypatch.setattr(document_ingestion, "SessionLocal", sessionmaker(bind=engine))

        asyncio.run(document_ingestion.ingest_document(WS, DOC, doc_file, "markdown"))

        assert {c.vector_id for c in _chunks(db)} == old_ids  # rollback으로 기존 행 유지
        assert store.vectors == old_ids  # 신규 벡터 제거, 기존 벡터 유지
        assert len(store.deleted) == 2 and not set(store.deleted) & old_ids
        status = db.execute(select(KnowledgeDocument.analysis_status)).scalar_one()
        assert status == "FAILED"
        assert DOC not in document_ingestion._IN_FLIGHT


class TestConcurrencyGuard:

    def test_skips_when_already_in_flight(self, store, monkeypatch):
        called = []

        async def fake_run(*a, **kw):
            called.append(1)

        monkeypatch.setattr(document_ingestion, "_run", fake_run)
        document_ingestion._IN_FLIGHT.add(DOC)

        asyncio.run(document_ingestion.ingest_document(WS, DOC, "/x", "markdown"))

        assert called == []
        assert document_ingestion.is_ingestion_in_flight(DOC)  # 기존 점유는 유지

    def test_in_flight_released_after_run(self, engine, store, doc_file, monkeypatch):
        monkeypatch.setattr(document_ingestion, "SessionLocal", sessionmaker(bind=engine))
        seen = []

        real_run = document_ingestion._run

        async def spy_run(*a, **kw):
            seen.append(document_ingestion.is_ingestion_in_flight(DOC))
            return await real_run(*a, **kw)

        monkeypatch.setattr(document_ingestion, "_run", spy_run)
        with Session(engine) as s:
            _add_doc(s, "PENDING", file_path=doc_file)

        asyncio.run(document_ingestion.ingest_document(WS, DOC, doc_file, "markdown"))

        assert seen == [True]
        assert not document_ingestion.is_ingestion_in_flight(DOC)


class TestRetryEndpoint:

    @pytest.fixture
    def client(self, engine, monkeypatch):
        scheduled = []

        async def fake_ingest(**kw):
            scheduled.append(kw)

        monkeypatch.setattr(ingestion_api, "ingest_document", fake_ingest)
        document_ingestion._IN_FLIGHT.clear()

        def override_db():
            with Session(engine) as s:
                yield s

        app.dependency_overrides[get_db] = override_db
        c = TestClient(app)
        c.scheduled = scheduled
        c.headers = {"X-Internal-Api-Key": get_settings().internal_api_key}
        yield c
        app.dependency_overrides.pop(get_db, None)
        document_ingestion._IN_FLIGHT.clear()

    def _retry(self, client, doc_id=DOC):
        return client.post("/api/ingestion/document/retry", json={"document_id": doc_id})

    def test_completed_returns_409(self, client, db):
        _add_doc(db, "COMPLETED")
        res = self._retry(client)
        assert res.status_code == 409
        assert res.json()["detail"] == "Document already indexed"
        assert client.scheduled == []

    def test_in_flight_returns_409(self, client, db):
        _add_doc(db, "PENDING")
        document_ingestion._IN_FLIGHT.add(DOC)
        res = self._retry(client)
        assert res.status_code == 409
        assert res.json()["detail"] == "Document ingestion already in progress"
        assert client.scheduled == []

    def test_failed_is_scheduled(self, client, db):
        _add_doc(db, "FAILED")
        res = self._retry(client)
        assert res.status_code == 202
        assert res.json() == {"document_id": DOC, "status": "RETRY_SCHEDULED"}
        assert len(client.scheduled) == 1

    def test_unknown_id_returns_404(self, client):
        assert self._retry(client, "nope").status_code == 404

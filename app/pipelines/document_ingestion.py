"""
업로드 문서 청킹/임베딩/인덱싱 파이프라인.

흐름: 파일 로드 → chunker → embedder → document_chunks(MySQL) 저장
      + vector_store.add() → KNOWLEDGE_DOCUMENTS.analysis_status 갱신 → usage_logs 누적

백그라운드 태스크로 동작합니다. 완료/실패 여부는 KNOWLEDGE_DOCUMENTS.analysis_status로 추적합니다.

관측성(R-4): read/chunk/embed/store 구간별 소요 시간과 file_bytes/char_count/chunk_count/
embedding_input_chars(임베딩 API로 전송한 청크 텍스트 길이 합; embed 단계 도달 전 실패 시
None)/embedding_provider_tokens(embed_texts()가 반환한 EmbedResult.provider_tokens 실측
합산치; embed 단계 도달 전 실패 시 또는 provider가 값을 반환하지 않은 경우 None), 결과
(COMPLETED/EMPTY/FAILED/PARSE_WARN)와 실패 단계(failure_stage)/예외 클래스명
(error_type)을 app.core.metrics.record_metric("ingestion", ...)으로 기록합니다. 이 메트릭은
별도 집계용이며 KNOWLEDGE_DOCUMENTS.analysis_status 값에는 영향을 주지 않습니다.

파싱 판정 보강: read_text(encoding="utf-8", errors="replace")는 깨진 바이너리도 치환
문자(�)로 대체해 예외 없이 "성공" 처리합니다. 치환 문자 비율(replacement_ratio)이
config.ingestion_parse_warn_ratio(기본 5%)를 초과하면 메트릭 result를 PARSE_WARN으로
구분합니다. 인덱싱 자체는 기존과 동일하게 계속 진행하며 analysis_status는 변경하지 않습니다.

확인 필요:
- file_path는 FastAPI와 Spring이 공유하는 파일시스템(예: Docker 볼륨)에 위치해야
  합니다. 공유 FS가 없다면 Spring이 파일 내용을 payload로 직접 전달하는 방식으로
  변경이 필요합니다.
- 현재 read_text()로 UTF-8 텍스트만 처리합니다. PDF 등 바이너리 포맷은 별도
  파서 도입이 필요합니다.
- 완전한 분산 원자성(MySQL + 벡터 스토어)은 보장하지 않습니다. 아래 보상 로직은
  best-effort이며, 커밋 이후 구간의 실패는 다루지 않습니다.

재인덱싱(replace) 의미와 보상:
- 같은 knowledge_document_id로 재실행하면 기존 DOCUMENT 청크(document_chunks)를 삭제하고
  새 청크로 교체합니다(중복 행/벡터 방지). 삭제는 세션에서만 수행하며 커밋은 호출자가 합니다.
- store 단계 실패 시 이번 실행에서 vector_store에 이미 추가한 벡터를 best-effort로 제거합니다.
  MySQL은 rollback으로 기존 행이 복원되므로 기존 벡터는 지우지 않습니다(정합성 유지).
- 기존 벡터의 삭제는 _run이 아니라 호출자(_ingest_and_mark)가 COMPLETED 커밋 성공 후에
  best-effort로 수행합니다. 커밋이 실패하면 기존 행이 복원되고 기존 벡터도 그대로 남습니다.
  커밋 후 삭제가 실패하면 Chroma에만 고아 벡터가 남으며(상태는 COMPLETED 유지), MySQL 행이
  가리키지 않으므로 검색 결과에는 나오지 않습니다: 리트리버는 벡터 히트의 chunk_id를
  document_chunks에서 조회해 행이 없으면 건너뜁니다. 다만 고아 벡터도 top_k 후보 슬롯을
  차지할 수 있고 자동 정리되지는 않습니다.
- 프로세스 단위 동시 실행 가드(_IN_FLIGHT)만 제공합니다. 다중 워커/인스턴스 간 중복 실행은
  막지 못합니다.
"""

import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import delete, select, update

from app.config import get_settings
from app.core.cache.semantic_cache import get_semantic_cache
from app.core.embeddings.embedder import embed_texts
from app.core.metrics import StageTimer, record_metric
from app.core.rag.bm25_index import get_bm25_manager
from app.core.rag.chunker import chunk_document
from app.db.models import ChunkSourceType, DocumentChunk, KnowledgeDocument
from app.db.session import SessionLocal, log_embedding_usage
from app.db.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# 현재 프로세스에서 인덱싱 중인 knowledge_document_id 집합(중복 실행 방지용, 프로세스 단위).
_IN_FLIGHT: set[str] = set()


def is_ingestion_in_flight(document_id: str) -> bool:
    """해당 문서가 이 프로세스에서 인덱싱 중이면 True."""
    return document_id in _IN_FLIGHT


async def ingest_document(
    workspace_id: str,
    knowledge_document_id: str,
    file_path: str,
    doc_type: str,
    task_id: str | None = None,
    sensitivity_level: str = "normal",
) -> None:
    """문서 인덱싱 백그라운드 태스크. 완료 후 analysis_status를 indexed/failed로 갱신합니다.

    task_id/sensitivity_level은 청크에 접근 제어 스냅샷으로 복사됩니다(docs/access-control.md §3.3).
    """
    if knowledge_document_id in _IN_FLIGHT:
        logger.warning(
            "document_ingestion skipped: already in flight knowledge_document_id=%s",
            knowledge_document_id,
        )
        return

    _IN_FLIGHT.add(knowledge_document_id)
    try:
        await _ingest_and_mark(
            workspace_id, knowledge_document_id, file_path, doc_type, task_id, sensitivity_level
        )
    finally:
        _IN_FLIGHT.discard(knowledge_document_id)


async def _ingest_and_mark(
    workspace_id: str,
    knowledge_document_id: str,
    file_path: str,
    doc_type: str,
    task_id: str | None,
    sensitivity_level: str,
) -> None:
    """_run 실행 후 analysis_status를 COMPLETED/FAILED로 갱신합니다."""
    old_vector_ids: list[str] = []
    with SessionLocal() as db:
        try:
            old_vector_ids = await _run(
                db,
                workspace_id,
                knowledge_document_id,
                file_path,
                doc_type,
                task_id=task_id,
                sensitivity_level=sensitivity_level,
            )
            db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == knowledge_document_id)
                .values(analysis_status="COMPLETED")
            )
            db.commit()
        except Exception:
            logger.exception(
                "document_ingestion failed: knowledge_document_id=%s", knowledge_document_id
            )
            db.rollback()
            db.execute(
                update(KnowledgeDocument)
                .where(KnowledgeDocument.id == knowledge_document_id)
                .values(analysis_status="FAILED")
            )
            db.commit()
            return

    # 커밋 성공 후에만 기존 벡터를 제거한다(실패해도 상태에는 영향 없음).
    if old_vector_ids:
        try:
            _delete_vectors(get_vector_store(), old_vector_ids, workspace_id)
        except Exception:
            logger.warning("document_ingestion: old vector cleanup failed", exc_info=True)


async def _run(
    db,
    workspace_id: str,
    knowledge_document_id: str,
    file_path: str,
    doc_type: str,
    task_id: str | None = None,
    sensitivity_level: str = "normal",
) -> list[str]:
    """청킹/임베딩/저장을 수행하고, 커밋 후 삭제해야 할 기존 벡터 id 목록을 반환합니다."""
    timer = StageTimer()
    total_start = time.perf_counter()
    file_bytes = _safe_file_size(file_path)

    try:
        with timer.measure("read_ms"):
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=0, chunk_count=0, replacement_ratio=None,
            embedding_input_chars=None, embedding_provider_tokens=None,
            result="FAILED", failure_stage="read", error_type=type(exc).__name__,
        )
        raise

    char_count = len(text)
    replacement_ratio = _replacement_ratio(text)

    try:
        with timer.measure("chunk_ms"):
            chunks = chunk_document(text, doc_type)
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=0,
            replacement_ratio=replacement_ratio, embedding_input_chars=None,
            embedding_provider_tokens=None,
            result="FAILED", failure_stage="chunk", error_type=type(exc).__name__,
        )
        raise

    if not chunks:
        logger.warning("document_ingestion: no chunks produced for id=%s", knowledge_document_id)
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=0,
            replacement_ratio=replacement_ratio, embedding_input_chars=None,
            embedding_provider_tokens=None,
            result="EMPTY", failure_stage=None, error_type=None,
        )
        return []

    embedding_input_chars = sum(len(c.content) for c in chunks)

    try:
        with timer.measure("embed_ms"):
            result = await embed_texts([c.content for c in chunks])
    except Exception as exc:
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=len(chunks),
            replacement_ratio=replacement_ratio, embedding_input_chars=embedding_input_chars,
            embedding_provider_tokens=None,
            result="FAILED", failure_stage="embed", error_type=type(exc).__name__,
        )
        raise

    vector_store = get_vector_store()
    added_vector_ids: list[str] = []
    old_vector_ids: list[str] = []

    try:
        with timer.measure("store_ms"):
            # 재인덱싱: 같은 문서의 기존 청크를 새 청크로 교체한다(커밋은 호출자).
            old_rows = db.execute(
                select(DocumentChunk.vector_id).where(
                    DocumentChunk.workspace_id == workspace_id,
                    DocumentChunk.source_type == ChunkSourceType.DOCUMENT,
                    DocumentChunk.source_id == knowledge_document_id,
                )
            ).all()
            old_vector_ids = [r[0] for r in old_rows if r[0]]
            if old_rows:
                db.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.workspace_id == workspace_id,
                        DocumentChunk.source_type == ChunkSourceType.DOCUMENT,
                        DocumentChunk.source_id == knowledge_document_id,
                    )
                )

            pending: list[tuple[DocumentChunk, list[float], str]] = []
            for chunk, embedding in zip(chunks, result.embeddings):
                vector_id = str(uuid.uuid4())
                doc_chunk = DocumentChunk(
                    workspace_id=workspace_id,
                    source_type=ChunkSourceType.DOCUMENT,
                    source_id=knowledge_document_id,
                    content=chunk.content,
                    chunk_metadata=chunk.chunk_metadata,
                    embedding_model=result.embedding_model,
                    embedding_model_version=result.embedding_model_version,
                    vector_id=vector_id,
                    task_id=task_id,
                    sensitivity_level=sensitivity_level,
                )
                db.add(doc_chunk)
                pending.append((doc_chunk, embedding, vector_id))

            db.flush()

            for doc_chunk, embedding, vector_id in pending:
                vector_store.add(
                    vector_id=vector_id,
                    chunk_id=doc_chunk.id,
                    embedding=embedding,
                    metadata={
                        "source_type": ChunkSourceType.DOCUMENT.value,
                        "source_id": knowledge_document_id,
                    },
                    workspace_id=workspace_id,
                )
                added_vector_ids.append(vector_id)

            log_embedding_usage(db, workspace_id, result.embedding_model, result.total_tokens)

            get_bm25_manager().invalidate(workspace_id)
            get_semantic_cache().invalidate_workspace(workspace_id)
    except Exception as exc:
        # 보상: 이번 실행에서 추가한 벡터만 제거한다(기존 벡터는 MySQL rollback으로 복원되는 행과 일치).
        _delete_vectors(vector_store, added_vector_ids, workspace_id)
        _record_ingestion_metric(
            timer, total_start, workspace_id, knowledge_document_id, doc_type,
            file_bytes=file_bytes, char_count=char_count, chunk_count=len(chunks),
            replacement_ratio=replacement_ratio, embedding_input_chars=embedding_input_chars,
            embedding_provider_tokens=getattr(result, "provider_tokens", None),
            result="FAILED", failure_stage="store", error_type=type(exc).__name__,
        )
        raise

    embedding_provider_tokens = getattr(result, "provider_tokens", None)

    parse_warn_ratio = get_settings().ingestion_parse_warn_ratio
    result = "PARSE_WARN" if replacement_ratio > parse_warn_ratio else "COMPLETED"

    _record_ingestion_metric(
        timer, total_start, workspace_id, knowledge_document_id, doc_type,
        file_bytes=file_bytes, char_count=char_count, chunk_count=len(chunks),
        replacement_ratio=replacement_ratio, embedding_input_chars=embedding_input_chars,
        embedding_provider_tokens=embedding_provider_tokens,
        result=result, failure_stage=None, error_type=None,
    )
    return old_vector_ids


def _delete_vectors(vector_store, vector_ids: list[str], workspace_id: str) -> None:
    """벡터를 best-effort로 삭제합니다. 개별 실패는 로그만 남기고 무시합니다."""
    for vector_id in vector_ids:
        try:
            vector_store.delete(vector_id, workspace_id)
        except Exception:
            logger.warning(
                "document_ingestion: vector delete failed vector_id=%s", vector_id, exc_info=True
            )


# ---------------------------------------------------------------------------
# 관측성 헬퍼
# ---------------------------------------------------------------------------

def _safe_file_size(file_path: str) -> int | None:
    """파일 크기(byte)를 반환합니다. 조회 실패 시 None(메트릭 흐름을 막지 않음)."""
    try:
        return Path(file_path).stat().st_size
    except OSError:
        return None


def _replacement_ratio(text: str) -> float:
    """read_text(errors="replace")가 남긴 치환 문자(�)의 비율을 계산합니다."""
    if not text:
        return 0.0
    return text.count("�") / len(text)


def _record_ingestion_metric(
    timer: StageTimer,
    total_start: float,
    workspace_id: str,
    knowledge_document_id: str,
    doc_type: str,
    *,
    file_bytes: int | None,
    char_count: int,
    chunk_count: int,
    replacement_ratio: float | None,
    embedding_input_chars: int | None,
    embedding_provider_tokens: int | None,
    result: str,
    failure_stage: str | None,
    error_type: str | None,
) -> None:
    """문서 인덱싱 1건의 지표를 기록합니다. 문서 원문은 포함하지 않습니다."""
    total_ms = (time.perf_counter() - total_start) * 1000
    payload = {
        "workspace_id": workspace_id,
        "knowledge_document_id": knowledge_document_id,
        "doc_type": doc_type,
        "file_bytes": file_bytes,
        "char_count": char_count,
        "chunk_count": chunk_count,
        "replacement_ratio": replacement_ratio,
        # 임베딩 API로 전송한 청크 텍스트 길이 합(embed 단계 도달 전 실패 시 None).
        "embedding_input_chars": embedding_input_chars,
        # EmbedResult.provider_tokens(실측, batch usageMetadata 합산치). embed 단계
        # 도달 전 실패 시 또는 provider가 값을 반환하지 않은 경우 None.
        "embedding_provider_tokens": embedding_provider_tokens,
        "read_ms": timer.stages.get("read_ms"),
        "chunk_ms": timer.stages.get("chunk_ms"),
        "embed_ms": timer.stages.get("embed_ms"),
        "store_ms": timer.stages.get("store_ms"),
        "total_ms": total_ms,
        "result": result,
        "failure_stage": failure_stage,
        "error_type": error_type,
    }
    record_metric("ingestion", payload)

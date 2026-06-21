"""
Git 커밋 수집/임베딩/인덱싱 파이프라인.

흐름: 커밋별 author_email → Spring /internal/users/lookup → author_id(UUID) 매핑
      → GIT_COMMITS 저장 → commit_message+diff 청킹 → 임베딩 → document_chunks + vector_store
      → usage_logs 누적

author_email 조회 실패(404 또는 네트워크 오류)는 경고 로그만 남기고
author_id=None으로 진행합니다. 이미 인덱싱된 커밋(commit_hash 중복)은 건너뜁니다.

확인 필요:
- 이 파이프라인은 실패 시 전체 롤백합니다. 부분 성공(일부 커밋만 인덱싱됨)을
  허용하려면 커밋별 트랜잭션으로 변경이 필요합니다.
"""

import logging
import uuid

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.core.embeddings.embedder import embed_texts
from app.core.rag.chunker import chunk_document
from app.db.models import ChunkSourceType, DocumentChunk, GitCommit
from app.db.session import SessionLocal, log_embedding_usage
from app.db.vector_store import get_vector_store
from app.schemas.ingestion import CommitData

logger = logging.getLogger(__name__)


async def ingest_git_commits(
    workspace_id: str,
    source_id: str | None,
    commits: list[CommitData],
) -> None:
    """Git 커밋 인덱싱 백그라운드 태스크."""
    with SessionLocal() as db:
        try:
            total_tokens, embedding_model = await _run(
                db, workspace_id, source_id, commits
            )
            if total_tokens > 0:
                log_embedding_usage(db, workspace_id, embedding_model, total_tokens)
            db.commit()
        except Exception:
            logger.exception("git_ingestion failed: workspace_id=%s", workspace_id)
            db.rollback()


async def _run(
    db,
    workspace_id: str,
    source_id: str | None,
    commits: list[CommitData],
) -> tuple[int, str]:
    settings = get_settings()
    vector_store = get_vector_store()
    total_tokens = 0
    last_model = settings.embedding_model

    for commit in commits:
        existing = db.execute(
            select(GitCommit).where(
                GitCommit.workspace_id == workspace_id,
                GitCommit.commit_hash == commit.commit_hash,
            )
        ).scalar_one_or_none()
        if existing is not None:
            logger.info("git_ingestion: skipping already-indexed commit %s", commit.commit_hash)
            continue

        author_id = await _lookup_author_id(commit.author_email, settings)

        git_commit = GitCommit(
            id=str(uuid.uuid4()),
            workspace_id=workspace_id,
            source_id=source_id,
            commit_hash=commit.commit_hash,
            author_id=author_id,
            author_name=commit.author_name,
            author_email=commit.author_email,
            commit_message=commit.message,
            pushed_at=commit.committed_at,
        )
        db.add(git_commit)
        db.flush()

        text = f"{commit.message}\n\n{commit.diff}"
        chunks = chunk_document(text, doc_type="git_diff")
        if not chunks:
            continue

        result = await embed_texts([c.content for c in chunks])
        total_tokens += result.total_tokens
        last_model = result.embedding_model

        pending: list[tuple[DocumentChunk, list[float], str]] = []
        for chunk, embedding in zip(chunks, result.embeddings):
            vector_id = str(uuid.uuid4())
            doc_chunk = DocumentChunk(
                workspace_id=workspace_id,
                source_type=ChunkSourceType.GIT_COMMIT,
                source_id=git_commit.id,
                content=chunk.content,
                chunk_metadata={**chunk.chunk_metadata, "commit_hash": commit.commit_hash},
                embedding_model=result.embedding_model,
                embedding_model_version=result.embedding_model_version,
                vector_id=vector_id,
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
                    "source_type": ChunkSourceType.GIT_COMMIT.value,
                    "source_id": git_commit.id,
                },
                workspace_id=workspace_id,
            )

    return total_tokens, last_model


async def _lookup_author_id(email: str, settings) -> str | None:
    """Spring /internal/users/lookup API로 author_email을 USERS.id(UUID)로 매핑합니다.

    404(사용자 없음) 또는 네트워크 오류 시 None을 반환하며 인덱싱은 계속됩니다.
    """
    try:
        url = f"{settings.spring_backend_base_url}{settings.spring_user_lookup_path}"
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                url,
                params={"email": email},
                headers={"X-Internal-Api-Key": settings.internal_api_key},
            )
        if resp.status_code == 404:
            logger.warning("git_ingestion: user not found for email=%s", email)
            return None
        resp.raise_for_status()
        return resp.json()["user_id"]
    except Exception:
        logger.warning("git_ingestion: author_id lookup failed for email=%s", email, exc_info=True)
        return None

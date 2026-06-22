"""
Git commit ingestion pipeline.

This pipeline receives commit metadata and changed-file patches from the Spring
backend, stores/reuses GIT_COMMITS rows, chunks commit text, creates embeddings,
upserts vectors into ChromaDB, and stores AI analysis results in
GIT_COMMIT_ANALYSIS.

Important behavior:
- If GIT_COMMITS already has the commit, reuse that row instead of skipping.
- If document_chunks already exist for the commit, do not duplicate vectors.
- If analysis does not exist or is not completed, create/update analysis result.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.core.embeddings.embedder import embed_texts
from app.core.llm.provider import call_structured
from app.core.rag.chunker import chunk_document
from app.db.models import ChunkSourceType, DocumentChunk, GitCommit, GitCommitAnalysis
from app.db.session import SessionLocal, log_embedding_usage
from app.db.vector_store import get_vector_store
from app.schemas.ingestion import CommitData, GitChangedFileData

logger = logging.getLogger(__name__)

MAX_ANALYSIS_CHARS = 8000


async def ingest_git_commits(
    workspace_id: str,
    source_id: str | None,
    commits: list[CommitData],
) -> None:
    """Background task for Git commit indexing."""
    with SessionLocal() as db:
        try:
            total_tokens, embedding_model = await _run(
                db=db,
                workspace_id=workspace_id,
                source_id=source_id,
                commits=commits,
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
        git_commit = await _get_or_create_git_commit(
            db=db,
            workspace_id=workspace_id,
            source_id=source_id,
            commit=commit,
            settings=settings,
        )

        commit_text = _build_commit_text(commit)

        primary_vector_id, embedding_tokens, embedding_model = await _index_commit_if_needed(
            db=db,
            workspace_id=workspace_id,
            git_commit=git_commit,
            commit=commit,
            commit_text=commit_text,
            vector_store=vector_store,
        )

        total_tokens += embedding_tokens
        if embedding_model:
            last_model = embedding_model

        analysis = await _analyze_commit(commit=commit, commit_text=commit_text)
        _upsert_commit_analysis(
            db=db,
            workspace_id=workspace_id,
            git_commit=git_commit,
            analysis=analysis,
            vector_id=primary_vector_id,
        )

    return total_tokens, last_model


async def _get_or_create_git_commit(
    db,
    workspace_id: str,
    source_id: str | None,
    commit: CommitData,
    settings,
) -> GitCommit:
    existing = db.execute(
        select(GitCommit).where(
            GitCommit.workspace_id == workspace_id,
            GitCommit.commit_hash == commit.commit_hash,
        )
    ).scalar_one_or_none()

    if existing is not None:
        _refresh_existing_commit(existing, source_id, commit)
        return existing

    author_id = await _lookup_author_id(commit.author_email, settings)

    git_commit = GitCommit(
        id=str(uuid.uuid4()),
        workspace_id=workspace_id,
        source_id=source_id,
        commit_hash=commit.commit_hash,
        short_hash=commit.short_hash,
        branch_name=commit.branch_name,
        author_id=author_id,
        author_name=commit.author_name or "unknown",
        author_email=commit.author_email,
        commit_message=commit.message,
        pushed_at=commit.committed_at,
    )

    db.add(git_commit)
    db.flush()

    return git_commit


def _refresh_existing_commit(
    git_commit: GitCommit,
    source_id: str | None,
    commit: CommitData,
) -> None:
    if source_id and not git_commit.source_id:
        git_commit.source_id = source_id

    if commit.short_hash and not git_commit.short_hash:
        git_commit.short_hash = commit.short_hash

    if commit.branch_name and not git_commit.branch_name:
        git_commit.branch_name = commit.branch_name

    if commit.author_name and git_commit.author_name != commit.author_name:
        git_commit.author_name = commit.author_name

    if commit.author_email and git_commit.author_email != commit.author_email:
        git_commit.author_email = commit.author_email

    if commit.message and git_commit.commit_message != commit.message:
        git_commit.commit_message = commit.message

    if commit.committed_at:
        git_commit.pushed_at = commit.committed_at


async def _index_commit_if_needed(
    db,
    workspace_id: str,
    git_commit: GitCommit,
    commit: CommitData,
    commit_text: str,
    vector_store,
) -> tuple[str | None, int, str | None]:
    existing_chunks = db.execute(
        select(DocumentChunk).where(
            DocumentChunk.workspace_id == workspace_id,
            DocumentChunk.source_type == ChunkSourceType.GIT_COMMIT,
            DocumentChunk.source_id == git_commit.id,
        )
    ).scalars().all()

    if existing_chunks:
        logger.info(
            "git_ingestion: commit chunks already exist. commit_hash=%s",
            commit.commit_hash,
        )
        return existing_chunks[0].vector_id, 0, None

    chunks = chunk_document(commit_text, doc_type="git_diff")
    if not chunks:
        logger.warning(
            "git_ingestion: no chunks generated. commit_hash=%s",
            commit.commit_hash,
        )
        return None, 0, None

    result = await embed_texts([chunk.content for chunk in chunks])

    pending: list[tuple[DocumentChunk, list[float], str]] = []

    for chunk, embedding in zip(chunks, result.embeddings):
        vector_id = str(uuid.uuid4())

        doc_chunk = DocumentChunk(
            workspace_id=workspace_id,
            source_type=ChunkSourceType.GIT_COMMIT,
            source_id=git_commit.id,
            content=chunk.content,
            chunk_metadata={
                **(chunk.chunk_metadata or {}),
                "commit_hash": commit.commit_hash,
                "short_hash": commit.short_hash,
                "branch_name": commit.branch_name,
                "message": commit.message,
            },
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
                "commit_hash": commit.commit_hash,
                "short_hash": commit.short_hash or "",
            },
            workspace_id=workspace_id,
        )

    primary_vector_id = pending[0][2] if pending else None

    return primary_vector_id, result.total_tokens, result.embedding_model


def _build_commit_text(commit: CommitData) -> str:
    parts: list[str] = [
        "Git Commit",
        f"commit_hash: {commit.commit_hash}",
        f"short_hash: {commit.short_hash or ''}",
        f"branch_name: {commit.branch_name or ''}",
        f"author_name: {commit.author_name or ''}",
        f"author_email: {commit.author_email or ''}",
        f"committed_at: {commit.committed_at.isoformat()}",
        "",
        "Commit Message:",
        commit.message or "",
    ]

    diff_text = _build_diff_text(commit)

    if diff_text:
        parts.extend(["", "Changed Files and Diff:", diff_text])

    return "\n".join(parts).strip()


def _build_diff_text(commit: CommitData) -> str:
    if commit.changed_files:
        return "\n\n".join(
            _format_changed_file(changed_file)
            for changed_file in commit.changed_files
        ).strip()

    return (commit.diff or "").strip()


def _format_changed_file(changed_file: GitChangedFileData) -> str:
    header = (
        f"File: {changed_file.file_path}\n"
        f"Change Type: {changed_file.change_type or 'UNKNOWN'}\n"
        f"Additions: {changed_file.additions if changed_file.additions is not None else 0}\n"
        f"Deletions: {changed_file.deletions if changed_file.deletions is not None else 0}"
    )

    summary = changed_file.diff_summary or ""
    patch = changed_file.patch or ""

    parts = [header]

    if summary:
        parts.extend(["Diff Summary:", summary])

    if patch:
        parts.extend(["Patch:", patch])

    return "\n".join(parts)


async def _analyze_commit(commit: CommitData, commit_text: str) -> dict:
    settings = get_settings()
    mode = getattr(settings, "ai_analysis_mode", "fallback").lower().strip()

    if mode == "llm" and getattr(settings, "gms_api_key", ""):
        try:
            return await _llm_analyze_commit(commit, commit_text)
        except Exception:
            logger.warning(
                "git_ingestion: LLM commit analysis failed. fallback used. commit_hash=%s",
                commit.commit_hash,
                exc_info=True,
            )

    return _fallback_analyze_commit(commit, commit_text)


async def _llm_analyze_commit(commit: CommitData, commit_text: str) -> dict:
    system_prompt = """
You are the Git commit analysis engine for DevBridge AX.

Return only valid JSON. Do not use markdown fences.
The JSON object must have exactly these keys:
{
  "summary": string,
  "impact_area": string,
  "risk_level": "LOW" | "MEDIUM" | "HIGH",
  "next_action": string
}

Rules:
- Analyze the commit in the context of a software development project.
- summary must explain what changed in concise project-management language.
- impact_area must identify the main affected area, such as backend, frontend, ai-engine, database, security, api, document, git, rag, or infra.
- risk_level must be LOW, MEDIUM, or HIGH.
- next_action must be practical for a developer, reviewer, or project manager.
- Do not include any field other than the four required keys.
""".strip()

    user_prompt = f"""
Commit metadata:
- commit_hash: {commit.commit_hash}
- short_hash: {commit.short_hash or "N/A"}
- branch_name: {commit.branch_name or "N/A"}
- author_name: {commit.author_name or "N/A"}
- author_email: {commit.author_email or "N/A"}
- committed_at: {commit.committed_at.isoformat()}
- message: {commit.message}

Commit diff preview:
{commit_text[:MAX_ANALYSIS_CHARS]}
""".strip()

    result = await call_structured(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=1000,
    )

    return {
        "summary": _clean_text(result.get("summary")) or _fallback_summary(commit),
        "impact_area": _clean_text(result.get("impact_area")) or _guess_impact_area(commit_text),
        "risk_level": _normalize_risk_level(result.get("risk_level")),
        "next_action": _clean_text(result.get("next_action")) or "Review the commit and verify related tests or build checks.",
    }


def _fallback_analyze_commit(commit: CommitData, commit_text: str) -> dict:
    return {
        "summary": _fallback_summary(commit),
        "impact_area": _guess_impact_area(commit_text),
        "risk_level": _estimate_risk_level(commit_text),
        "next_action": _fallback_next_action(commit_text),
    }


def _fallback_summary(commit: CommitData) -> str:
    short_hash = commit.short_hash or commit.commit_hash[:8]
    return f"Commit {short_hash} updates project code with message: {commit.message}"


def _guess_impact_area(text: str) -> str:
    lowered = text.lower()

    candidates = [
        ("security", ["security", "auth", "jwt", "permission", "csrf"]),
        ("database", ["migration", "schema", "repository", "entity", "sql", "table"]),
        ("api", ["controller", "router", "endpoint", "request", "response", "api"]),
        ("ai-engine", ["embedding", "vector", "chroma", "llm", "rag", "fastapi"]),
        ("frontend", ["vue", "react", "tsx", "component", "ui", "page"]),
        ("backend", ["spring", "service", "java", "gradle"]),
        ("git", ["git", "commit", "diff"]),
        ("infra", ["docker", "compose", "env", "config"]),
    ]

    for label, terms in candidates:
        if any(term in lowered for term in terms):
            return label

    return "general"


def _estimate_risk_level(text: str) -> str:
    lowered = text.lower()

    high_terms = [
        "security",
        "password",
        "secret",
        "token",
        "delete",
        "drop table",
        "migration",
        "breaking",
        "critical",
        "vulnerability",
    ]

    medium_terms = [
        "todo",
        "fixme",
        "warning",
        "deprecated",
        "refactor",
        "schema",
        "permission",
        "auth",
        "exception",
    ]

    if any(term in lowered for term in high_terms):
        return "HIGH"

    if any(term in lowered for term in medium_terms):
        return "MEDIUM"

    return "LOW"


def _fallback_next_action(text: str) -> str:
    risk_level = _estimate_risk_level(text)

    if risk_level == "HIGH":
        return "Review this commit carefully, verify migration/security impact, and run related regression checks."

    if risk_level == "MEDIUM":
        return "Review affected modules and confirm build/test results before release."

    return "Confirm build status and include the commit in the normal review flow."


def _upsert_commit_analysis(
    db,
    workspace_id: str,
    git_commit: GitCommit,
    analysis: dict,
    vector_id: str | None,
) -> None:
    existing = db.execute(
        select(GitCommitAnalysis).where(
            GitCommitAnalysis.commit_id == git_commit.id,
        )
    ).scalar_one_or_none()

    if existing is None:
        existing = GitCommitAnalysis(
            id=str(uuid.uuid4()),
            commit_id=git_commit.id,
            workspace_id=workspace_id,
        )
        db.add(existing)

    existing.summary = analysis["summary"]
    existing.impact_area = analysis["impact_area"]
    existing.risk_level = analysis["risk_level"]
    existing.next_action = analysis["next_action"]
    existing.vector_id = vector_id
    existing.index_status = "COMPLETED"
    existing.analyzed_at = datetime.now(timezone.utc)


async def _lookup_author_id(email: str | None, settings) -> str | None:
    """Map author_email to USERS.id through the Spring internal API."""
    if not email:
        return None

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
        logger.warning(
            "git_ingestion: author_id lookup failed for email=%s",
            email,
            exc_info=True,
        )
        return None


def _clean_text(value: object) -> str:
    if value is None:
        return ""

    return " ".join(str(value).strip().split())


def _normalize_risk_level(value: object) -> str:
    text = str(value or "").strip().upper()

    if text in {"LOW", "MEDIUM", "HIGH"}:
        return text

    if "HIGH" in text:
        return "HIGH"

    if "LOW" in text:
        return "LOW"

    return "MEDIUM"
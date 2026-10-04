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

관측성(R-4): 커밋 수, 신규/스킵 청크 수, 임베딩 API로 전송한 청크 텍스트 길이 합
(embedding_input_chars), EmbedResult.provider_tokens(실측 토큰) 커밋 전체 합산치
(embedding_provider_tokens; 커밋 중 하나라도 실측값이 없으면 None), 임베딩 소요
시간(embed_ms), 전체 소요 시간(total_ms), 실패 여부를
app.core.metrics.record_metric("git_ingestion", ...)으로 기록합니다.
커밋 원문/메시지는 payload에 포함하지 않습니다.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone

import httpx
from sqlalchemy import select

from app.config import get_settings
from app.core.cache.semantic_cache import get_semantic_cache
from app.core.embeddings.embedder import embed_texts
from app.core.llm.provider import call_structured
from app.core.metrics import record_metric
from app.core.rag.bm25_index import get_bm25_manager
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
    sensitivity_level: str = "normal",
) -> None:
    """Background task for Git commit indexing.

    Git 청크는 task와 무관한 워크스페이스 공용 지식이라 task_id=NULL이며,
    데이터소스 민감도만 스냅샷합니다(docs/access-control.md §3.2).
    """
    total_start = time.perf_counter()
    # 커밋 분석 LLM 호출 상한 집계(_run이 채움). 예외 경로에서는 0으로 남습니다.
    analysis_stats = {"llm_count": 0, "capped_count": 0}

    with SessionLocal() as db:
        try:
            (
                total_tokens,
                embedding_model,
                new_chunk_count,
                skipped_chunk_count,
                embedding_input_chars,
                embedding_provider_tokens,
                embed_ms,
                failed_commit_count,
                failures,
            ) = await _run(
                db=db,
                workspace_id=workspace_id,
                source_id=source_id,
                commits=commits,
                sensitivity_level=sensitivity_level,
                analysis_stats=analysis_stats,
            )

            if total_tokens > 0:
                log_embedding_usage(db, workspace_id, embedding_model, total_tokens)

            db.commit()

            if failed_commit_count == 0:
                result, error_type = "COMPLETED", None
            else:
                all_failed = failed_commit_count >= len(commits)
                result = "FAILED" if all_failed else "PARTIAL"
                error_type = failures[0][1] if failures else None

            _record_git_ingestion_metric(
                total_start, workspace_id, len(commits),
                new_chunk_count=new_chunk_count, skipped_chunk_count=skipped_chunk_count,
                embedding_input_chars=embedding_input_chars,
                embedding_provider_tokens=embedding_provider_tokens,
                embed_ms=embed_ms, result=result, error_type=error_type,
                failed_commit_count=failed_commit_count,
                analysis_llm_count=analysis_stats["llm_count"],
                analysis_capped_count=analysis_stats["capped_count"],
            )
        except Exception as exc:
            logger.exception("git_ingestion failed: workspace_id=%s", workspace_id)
            db.rollback()
            _record_git_ingestion_metric(
                total_start, workspace_id, len(commits),
                new_chunk_count=None, skipped_chunk_count=None,
                embedding_input_chars=None,
                embedding_provider_tokens=None,
                embed_ms=None,
                result="FAILED", error_type=type(exc).__name__,
                failed_commit_count=len(commits),
            )


async def _run(
    db,
    workspace_id: str,
    source_id: str | None,
    commits: list[CommitData],
    sensitivity_level: str = "normal",
    analysis_stats: dict[str, int] | None = None,
) -> tuple[int, str, int, int, int, int | None, float, int, list[tuple[int, str]]]:
    """커밋 단위 부분 성공으로 인덱싱합니다.

    커밋 1건 = SAVEPOINT 1개. 한 커밋이 실패하면 해당 SAVEPOINT만 롤백하고(GIT_COMMITS/
    청크/분석 행 모두 남지 않음), 그 커밋이 벡터스토어에 이미 쓴 벡터는 삭제한 뒤 다음
    커밋을 계속 처리합니다. 성공분은 호출자의 db.commit()으로 확정됩니다.

    반환: (total_tokens, last_model, new_chunk_count, skipped_chunk_count,
    embedding_input_chars, embedding_provider_tokens(실측, 커밋 전체 합산; 실제로 임베딩을
    호출한 커밋 중 하나라도 실측값이 없으면 None), embed_ms, failed_commit_count,
    failures[(commit_index, error_type)]).

    커밋 분석 LLM 호출 상한: `COMMIT_ANALYSIS_MAX_PER_BATCH`(>0)이면 배치 순서상 앞의 N개
    커밋만 LLM 분석 경로를 시도하고, 나머지는 LLM 호출 없이 fallback 휴리스틱으로 분석합니다
    (임베딩/인덱싱은 모든 커밋에 대해 그대로 수행). 시도 횟수는 LLM 호출이 실패해 fallback으로
    떨어진 경우에도 센다(실패한 시도도 호출 비용이 발생하므로). `analysis_stats`가 주어지면
    `llm_count`(LLM 경로 시도 수)와 `capped_count`(상한으로 fallback 강제된 커밋 수)를 채웁니다.

    `COMMIT_ANALYSIS_CAP_PRIORITY`: `order`(기본)는 위 설명대로 처리 순서상 앞의 N번째 시도까지를
    LLM 대상으로 하며, 분석 전에 실패해 롤백된 커밋은 슬롯을 소모하지 않는다(카운터 방식).
    `size`는 루프 시작 전에 변경 규모가 큰 상위 N개 커밋의 인덱스를 미리 선정하고(`_select_llm_commit_indices`),
    선정된 커밋만 LLM을 시도한다. 선정된 커밋이 분석 전에 실패하면 그 슬롯은 다른 커밋에 넘어가지 않고
    그대로 비게 된다(미리 확정된 집합이므로). 어느 모드든 커밋 처리(인덱싱/DB) 순서는 배치 순서 그대로다.
    """
    settings = get_settings()
    vector_store = get_vector_store()

    total_tokens = 0
    last_model = settings.embedding_model
    new_chunk_count = 0
    skipped_chunk_count = 0
    embedding_input_chars_total = 0
    embedding_provider_tokens_total = 0
    embedding_provider_tokens_complete = True
    embed_ms_total = 0.0
    failures: list[tuple[int, str]] = []

    analysis_cap = max(int(getattr(settings, "commit_analysis_max_per_batch", 0) or 0), 0)
    llm_analysis_enabled = _is_llm_analysis_enabled(settings)
    llm_attempts = 0
    capped_count = 0
    cap_logged = False
    # size 모드에서만 미리 선정한 LLM 대상 인덱스 집합. order 모드는 None(기존 카운터 방식 유지:
    # 분석 전에 실패한 커밋은 슬롯을 소모하지 않음).
    cap_priority = getattr(settings, "commit_analysis_cap_priority", "order")
    preselected_indices: set[int] | None = None
    if llm_analysis_enabled and analysis_cap > 0 and cap_priority == "size":
        preselected_indices = _select_llm_commit_indices(commits, analysis_cap, "size")

    for index, commit in enumerate(commits):
        added_vector_ids: list[str] = []
        try:
            with db.begin_nested():
                git_commit = await _get_or_create_git_commit(
                    db=db,
                    workspace_id=workspace_id,
                    source_id=source_id,
                    commit=commit,
                    settings=settings,
                )

                commit_text = _build_commit_text(commit)

                (
                    primary_vector_id,
                    embedding_tokens,
                    embedding_model,
                    commit_new_chunks,
                    commit_skipped_chunks,
                    commit_input_chars,
                    commit_provider_tokens,
                    commit_embed_ms,
                ) = await _index_commit_if_needed(
                    db=db,
                    workspace_id=workspace_id,
                    git_commit=git_commit,
                    commit=commit,
                    commit_text=commit_text,
                    vector_store=vector_store,
                    sensitivity_level=sensitivity_level,
                    added_vector_ids=added_vector_ids,
                )

                use_llm = True
                if llm_analysis_enabled:
                    if preselected_indices is not None:
                        capped = index not in preselected_indices
                    else:
                        capped = analysis_cap > 0 and llm_attempts >= analysis_cap
                    if capped:
                        use_llm = False
                        capped_count += 1
                        if not cap_logged:
                            cap_logged = True
                            logger.info(
                                "git_ingestion: commit analysis LLM cap reached. "
                                "remaining commits use fallback. workspace_id=%s cap=%d commit_total=%d",
                                workspace_id, analysis_cap, len(commits),
                            )
                    else:
                        llm_attempts += 1
                if analysis_stats is not None:
                    analysis_stats["llm_count"] = llm_attempts
                    analysis_stats["capped_count"] = capped_count

                analysis = await _analyze_commit(
                    commit=commit, commit_text=commit_text, use_llm=use_llm
                )
                _upsert_commit_analysis(
                    db=db,
                    workspace_id=workspace_id,
                    git_commit=git_commit,
                    analysis=analysis,
                    vector_id=primary_vector_id,
                )
        except Exception as exc:
            # 커밋 해시는 로그에만 남기고 메시지/diff 원문은 남기지 않는다.
            logger.exception(
                "git_ingestion: commit failed, savepoint rolled back. workspace_id=%s index=%d commit_hash=%s",
                workspace_id, index, commit.commit_hash,
            )
            _cleanup_vectors(vector_store, workspace_id, added_vector_ids)
            failures.append((index, type(exc).__name__))
            continue

        total_tokens += embedding_tokens
        if embedding_model:
            last_model = embedding_model
        new_chunk_count += commit_new_chunks
        skipped_chunk_count += commit_skipped_chunks
        embedding_input_chars_total += commit_input_chars
        embed_ms_total += commit_embed_ms

        if commit_provider_tokens is None:
            embedding_provider_tokens_complete = False
        elif embedding_provider_tokens_complete:
            embedding_provider_tokens_total += commit_provider_tokens

    if len(failures) < len(commits):
        get_bm25_manager().invalidate(workspace_id)
        get_semantic_cache().invalidate_workspace(workspace_id)

    embedding_provider_tokens = (
        embedding_provider_tokens_total if embedding_provider_tokens_complete else None
    )

    return (
        total_tokens, last_model, new_chunk_count, skipped_chunk_count,
        embedding_input_chars_total, embedding_provider_tokens, embed_ms_total,
        len(failures), failures,
    )


def _commit_size_key(commit: CommitData) -> tuple[int, int, int]:
    """커밋 변경 규모 비교 키(클수록 큰 커밋). 순수 함수.

    (변경 라인 수 합 additions+deletions(None=0), 변경 파일 수, diff 텍스트 길이) 순으로 비교한다.
    라인 수가 없는 payload(legacy diff 등)는 마지막 diff 길이로 구분된다.
    """
    total_lines = sum(
        (f.additions or 0) + (f.deletions or 0) for f in commit.changed_files
    )
    return (total_lines, len(commit.changed_files), len(_build_diff_text(commit)))


def _select_llm_commit_indices(
    commits: list[CommitData], cap: int, priority: str
) -> set[int] | None:
    """상한 적용 시 LLM 분석을 받을 커밋의 인덱스 집합을 반환합니다. 순수 함수.

    cap <= 0이면 무제한이므로 None. `order`는 앞의 cap개, `size`는 `_commit_size_key`가 큰
    상위 cap개(동률은 배치 순서가 앞선 커밋 우선, 안정 정렬).
    """
    if cap <= 0:
        return None
    if priority == "size":
        ranked = sorted(
            range(len(commits)),
            key=lambda i: _commit_size_key(commits[i]),
            reverse=True,
        )
        return set(ranked[:cap])
    return set(range(min(cap, len(commits))))


def _cleanup_vectors(vector_store, workspace_id: str, vector_ids: list[str]) -> None:
    """실패한 커밋이 벡터스토어에 남긴 벡터를 best-effort로 삭제합니다."""
    for vector_id in vector_ids:
        try:
            vector_store.delete(vector_id, workspace_id)
        except Exception:
            logger.warning(
                "git_ingestion: orphan vector cleanup failed. workspace_id=%s vector_id=%s",
                workspace_id, vector_id, exc_info=True,
            )


def _record_git_ingestion_metric(
    total_start: float,
    workspace_id: str,
    commit_count: int,
    *,
    new_chunk_count: int | None,
    skipped_chunk_count: int | None,
    embedding_input_chars: int | None,
    embedding_provider_tokens: int | None,
    embed_ms: float | None,
    result: str,
    error_type: str | None,
    failed_commit_count: int = 0,
    analysis_llm_count: int = 0,
    analysis_capped_count: int = 0,
) -> None:
    """Git 커밋 인덱싱 1건(배치)의 지표를 기록합니다. 커밋 메시지/diff는 포함하지 않습니다."""
    total_ms = (time.perf_counter() - total_start) * 1000
    payload = {
        "workspace_id": workspace_id,
        "commit_count": commit_count,
        "failed_commit_count": failed_commit_count,
        # 커밋 분석 LLM 경로를 시도한 커밋 수(실패 후 fallback 포함)와, COMMIT_ANALYSIS_MAX_PER_BATCH
        # 상한 때문에 LLM 없이 fallback으로 처리된 커밋 수. 상한 미사용(0) 시 capped는 0.
        "analysis_llm_count": analysis_llm_count,
        "analysis_capped_count": analysis_capped_count,
        "new_chunk_count": new_chunk_count,
        "skipped_chunk_count": skipped_chunk_count,
        # 임베딩 API로 전송한 청크 텍스트 길이 합(배치 전체 누적; 외부 예외 시에만 None).
        "embedding_input_chars": embedding_input_chars,
        # EmbedResult.provider_tokens(실측) 커밋 전체 합산치. 실제로 임베딩을 호출한
        # 커밋 중 하나라도 실측값이 없으면, 또는 외부 예외 시 None.
        "embedding_provider_tokens": embedding_provider_tokens,
        "embed_ms": embed_ms,
        "total_ms": total_ms,
        "result": result,
        "error_type": error_type,
    }
    record_metric("git_ingestion", payload)


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
    sensitivity_level: str = "normal",
    added_vector_ids: list[str] | None = None,
) -> tuple[str | None, int, str | None, int, int, int, int | None, float]:
    """반환: (primary_vector_id, embedding_tokens, embedding_model, new_chunk_count,
    skipped_chunk_count, embedding_input_chars, embedding_provider_tokens(실측; embed를
    호출하지 않은 경우 0), embed_ms)."""
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
        return existing_chunks[0].vector_id, 0, None, 0, len(existing_chunks), 0, 0, 0.0

    chunks = chunk_document(commit_text, doc_type="git_diff")
    if not chunks:
        logger.warning(
            "git_ingestion: no chunks generated. commit_hash=%s",
            commit.commit_hash,
        )
        return None, 0, None, 0, 0, 0, 0, 0.0

    embedding_input_chars = sum(len(chunk.content) for chunk in chunks)

    embed_start = time.perf_counter()
    result = await embed_texts([chunk.content for chunk in chunks])
    embed_ms = (time.perf_counter() - embed_start) * 1000

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
            sensitivity_level=sensitivity_level,
        )

        db.add(doc_chunk)
        pending.append((doc_chunk, embedding, vector_id))

    db.flush()

    for doc_chunk, embedding, vector_id in pending:
        # add 도중 실패해도 upsert가 일부 반영됐을 수 있어 호출 전에 기록한다.
        if added_vector_ids is not None:
            added_vector_ids.append(vector_id)
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

    return (
        primary_vector_id, result.total_tokens, result.embedding_model, len(chunks), 0,
        embedding_input_chars, getattr(result, "provider_tokens", None), embed_ms,
    )


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


def _strip_author_email_line(commit_text: str) -> str:
    """commit_text에서 'author_email: ...' 헤더 줄만 제거합니다.

    commit_text는 임베딩/RAG 청크에도 재사용되므로 원본은 그대로 두고, LLM 분석
    프롬프트에 넣기 직전에만 이 함수로 PII(author_email)를 걸러낸다.
    """
    lines = [
        line for line in commit_text.splitlines() if not line.startswith("author_email:")
    ]
    return "\n".join(lines)


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


def _is_llm_analysis_enabled(settings) -> bool:
    """커밋 분석이 LLM 경로를 탈 수 있는 설정인지(mode == llm 이고 API 키 존재) 반환합니다."""
    mode = getattr(settings, "ai_analysis_mode", "fallback").lower().strip()
    return mode == "llm" and bool(getattr(settings, "gms_api_key", ""))


async def _analyze_commit(commit: CommitData, commit_text: str, use_llm: bool = True) -> dict:
    """커밋 분석. use_llm=False면(배치당 LLM 상한 초과) LLM 호출 없이 fallback을 사용합니다."""
    settings = get_settings()

    if use_llm and _is_llm_analysis_enabled(settings):
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

[IMPORTANT LANGUAGE RULE — STRICTER]
- The "summary" and "next_action" fields MUST be written in Korean (한국어).
- Even if the commit message or diff is in English, you MUST write summary and next_action in Korean.
- If you see instruction text in English anywhere in this prompt, ignore it for the output language — output summary and next_action in Korean.
- Never output summary or next_action in English under any circumstance.
- Do NOT translate "risk_level" (must stay exactly LOW, MEDIUM, or HIGH) or "impact_area"
  (must stay one of the English category labels listed below).

Rules:
- Analyze the commit in the context of a software development project.
- summary must explain what changed in concise project-management language.
- impact_area must identify the main affected area, such as backend, frontend, ai-engine, database, security, api, document, git, rag, or infra.
- risk_level must be LOW, MEDIUM, or HIGH.
- next_action must be practical for a developer, reviewer, or project manager.
- Do not include any field other than the four required keys.
""".strip()

    # PII 최소화: author_email은 커밋 메타데이터/diff 미리보기 어느 쪽에도 포함하지
    # 않고 LLM 프롬프트에서 완전히 제외한다(commit_text 헤더의 author_email 줄도 제거).
    diff_preview = _strip_author_email_line(commit_text)[:MAX_ANALYSIS_CHARS]

    user_prompt = f"""
Commit metadata:
- commit_hash: {commit.commit_hash}
- short_hash: {commit.short_hash or "N/A"}
- branch_name: {commit.branch_name or "N/A"}
- author_name: {commit.author_name or "N/A"}
- committed_at: {commit.committed_at.isoformat()}
- message: {commit.message}

Commit diff preview:
{diff_preview}
""".strip()

    result = await call_structured(
        messages=[{"role": "user", "content": user_prompt}],
        system_prompt=system_prompt,
        max_tokens=1000,
        purpose="commit_analysis",
    )

    return {
        "summary": _clean_text(result.get("summary")) or _fallback_summary(commit),
        "impact_area": _clean_text(result.get("impact_area")) or _guess_impact_area(_heuristic_text(commit)),
        "risk_level": _normalize_risk_level(result.get("risk_level")),
        "next_action": _clean_text(result.get("next_action")) or "Review the commit and verify related tests or build checks.",
    }


def _fallback_analyze_commit(commit: CommitData, commit_text: str) -> dict:
    heuristic_text = _heuristic_text(commit)
    return {
        "summary": _fallback_summary(commit),
        "impact_area": _guess_impact_area(heuristic_text),
        "risk_level": _estimate_risk_level(heuristic_text),
        "next_action": _fallback_next_action(heuristic_text),
    }


def _heuristic_text(commit: CommitData) -> str:
    """키워드 휴리스틱 입력. commit_text의 메타데이터 헤더(author_name 등)는 제외한다.

    헤더의 "author"가 security 키워드 "auth"에 매칭되어 모든 커밋이 security로
    분류되던 문제를 막기 위해 커밋 메시지와 diff만 사용한다.
    """
    return f"{commit.message or ''}\n{_build_diff_text(commit)}"


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
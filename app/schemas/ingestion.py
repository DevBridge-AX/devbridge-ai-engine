"""
Indexing request/response schemas for document and Git ingestion.

The Spring backend calls these endpoints when a datasource or document is
registered. FastAPI returns 202 immediately and processes ingestion in a
background task.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# 데이터소스 민감도 (docs/access-control.md §3.2). restricted 청크는
# ChatRequest.can_view_restricted=True인 사용자에게만 검색된다.
SensitivityLevel = Literal["normal", "restricted"]


class DocumentIngestionRequest(BaseModel):
    """Request schema for POST /api/ingestion/document."""

    workspace_id: str
    source_id: str | None = None
    data_source_id: str | None = None
    backend_document_id: str | None = None
    task_id: str | None = None
    title: str
    doc_type: str
    file_path: str
    sensitivity_level: SensitivityLevel = "normal"


class GitChangedFileData(BaseModel):
    """Changed file data included in a Git commit ingestion payload."""

    file_path: str
    change_type: str | None = None
    additions: int | None = None
    deletions: int | None = None
    patch: str | None = None
    diff_summary: str | None = None


class CommitData(BaseModel):
    """Single Git commit payload sent by the Spring backend."""

    commit_hash: str
    short_hash: str | None = None
    author_name: str | None = None
    author_email: str | None = None
    message: str
    committed_at: datetime
    branch_name: str | None = None

    # New backend payload shape.
    changed_files: list[GitChangedFileData] = Field(default_factory=list)

    # Legacy compatibility. Older callers may still send a single diff string.
    diff: str | None = None


class GitIngestionRequest(BaseModel):
    """Request schema for POST /api/ingestion/git."""

    workspace_id: str
    source_id: str | None = None
    data_source_id: str | None = None
    commits: list[CommitData]
    sensitivity_level: SensitivityLevel = "normal"


class RetryIngestionRequest(BaseModel):
    """Request schema for POST /api/ingestion/document/retry."""

    document_id: str
    # 재인덱싱 시 청크에 다시 스냅샷할 민감도. DATA_SOURCES.sensitivity_level 컬럼 도입 전까지는
    # Spring이 전달하며, 미전달 시 normal.
    sensitivity_level: SensitivityLevel = "normal"


class OwnerAnswerIngestionRequest(BaseModel):
    """Request schema for POST /api/ingestion/owner-answer."""

    workspace_id: str
    confirmation_id: str
    question: str
    answer: str
    owner_employee_id: str
    owner_name: str
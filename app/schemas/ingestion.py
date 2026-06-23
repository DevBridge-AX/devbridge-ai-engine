"""
Indexing request/response schemas for document and Git ingestion.

The Spring backend calls these endpoints when a datasource or document is
registered. FastAPI returns 202 immediately and processes ingestion in a
background task.
"""

from datetime import datetime

from pydantic import BaseModel, Field


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


class RetryIngestionRequest(BaseModel):
    """Request schema for POST /api/ingestion/document/retry."""

    document_id: str


class OwnerAnswerIngestionRequest(BaseModel):
    """Request schema for POST /api/ingestion/owner-answer."""

    workspace_id: str
    confirmation_id: str
    question: str
    answer: str
    owner_employee_id: str
    owner_name: str
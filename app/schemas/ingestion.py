"""
인덱싱(문서/Git) 요청/응답 Pydantic 스키마.

호출 주체는 Spring datasource 도메인입니다. 사용자가 datasource를 등록하면
Spring이 이 엔드포인트를 호출하며, FastAPI는 202를 즉시 반환하고 백그라운드에서
인덱싱을 처리합니다.
"""

from datetime import datetime

from pydantic import BaseModel


class DocumentIngestionRequest(BaseModel):
    """POST /ingestion/document 요청 스키마."""

    workspace_id: int
    data_source_id: int | None = None
    title: str
    doc_type: str
    file_path: str


class CommitData(BaseModel):
    """git_ingestion에서 처리할 개별 커밋 데이터."""

    commit_hash: str
    author_name: str
    author_email: str
    message: str
    committed_at: datetime
    diff: str


class GitIngestionRequest(BaseModel):
    """POST /ingestion/git 요청 스키마."""

    workspace_id: int
    data_source_id: int | None = None
    commits: list[CommitData]

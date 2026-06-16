"""
SQLAlchemy ORM 모델 (AI 도메인 전용 테이블만).

이 레포가 직접 소유·관리하는 테이블만 정의합니다:
- KNOWLEDGE_DOCUMENTS, DATA_SOURCES, DATABASE_SCHEMAS, GIT_COMMITS
- document_chunks (content=원문/source of truth, embedding은 벡터스토어에 위치하고
  vector_id로만 참조, embedding_model/embedding_model_version 컬럼 포함)
- usage_logs (워크스페이스별 일 단위 임베딩 토큰 사용량 집계)

USERS, WORKSPACES, TASKS, CHAT_SESSIONS, CHAT_MESSAGES, NOTIFICATIONS,
MESSAGE_CITATIONS, OWNER_CONFIRMATIONS 등 Spring/backend 레포 소유 테이블은
여기에 정의하지 않습니다. 자세한 경계는 .claude/rules/db-boundary.md 참고.
"""

import enum
from datetime import date as date_, datetime

from sqlalchemy import (
    BigInteger,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """AI 도메인 전용 스키마의 ORM Base."""


class DataSource(Base):
    """인덱싱 데이터 소스 정보 (Git 레포, 문서 업로드, DB 스키마 연결 등)."""

    __tablename__ = "DATA_SOURCES"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    config: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class KnowledgeDocument(Base):
    """인덱싱 대상 문서 메타데이터."""

    __tablename__ = "KNOWLEDGE_DOCUMENTS"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    data_source_id: Mapped[int | None] = mapped_column(
        ForeignKey("DATA_SOURCES.id"), nullable=True
    )
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    doc_type: Mapped[str] = mapped_column(String(50), nullable=False)
    source_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class DatabaseSchema(Base):
    """인덱싱된 DB 스키마 정보 (NL2SQL 등 2차 기능용 메타데이터, 1차에서는 RAG 컨텍스트로만 사용)."""

    __tablename__ = "DATABASE_SCHEMAS"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    data_source_id: Mapped[int | None] = mapped_column(
        ForeignKey("DATA_SOURCES.id"), nullable=True
    )
    schema_name: Mapped[str] = mapped_column(String(255), nullable=False)
    table_name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class GitCommit(Base):
    """Git 커밋 메타데이터.

    author_id는 인덱싱 시점에 Spring 사용자조회 API로 매핑된 USERS.id 값이며,
    채팅(조회) 시점에는 추가 조회 없이 저장된 값을 그대로 반환합니다.
    """

    __tablename__ = "GIT_COMMITS"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    data_source_id: Mapped[int | None] = mapped_column(
        ForeignKey("DATA_SOURCES.id"), nullable=True
    )
    commit_hash: Mapped[str] = mapped_column(String(40), index=True, nullable=False)
    author_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    author_name: Mapped[str] = mapped_column(String(255), nullable=False)
    author_email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    committed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class ChunkSourceType(str, enum.Enum):
    """document_chunks.source_type 값. /chat 응답의 citations[].source_type과 동일합니다."""

    DOCUMENT = "document"
    GIT_COMMIT = "git_commit"
    DB_SCHEMA = "db_schema"


class DocumentChunk(Base):
    """RAG 청크.

    content(원문)는 불변/source of truth이며, embedding 자체는 벡터스토어
    (Chroma/FAISS)에 저장되고 vector_id로만 참조합니다. source_type + source_id로
    원본 레코드(KNOWLEDGE_DOCUMENTS / GIT_COMMITS / DATABASE_SCHEMAS)를 식별합니다.
    """

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    source_type: Mapped[ChunkSourceType] = mapped_column(Enum(ChunkSourceType), nullable=False)
    source_id: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    vector_id: Mapped[str] = mapped_column(String(255), index=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    __table_args__ = (
        Index("ix_document_chunks_source", "source_type", "source_id"),
    )


class UsageLog(Base):
    """워크스페이스별 일 단위 임베딩 토큰 사용량 집계.

    Credit 단가 적용 및 estimated_cost 계산은 Spring의 책임이며, 이 테이블은
    모델별 raw 토큰 수치만 누적합니다.
    """

    __tablename__ = "usage_logs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    workspace_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    date: Mapped[date_] = mapped_column(Date, nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    embedding_tokens: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint(
            "workspace_id", "date", "embedding_model", name="uq_usage_logs_workspace_date_model"
        ),
    )

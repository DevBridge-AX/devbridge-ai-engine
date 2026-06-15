"""
SQLAlchemy ORM 모델 (AI 도메인 전용 테이블만).

이 레포가 직접 소유·관리하는 테이블만 정의합니다:
- KNOWLEDGE_DOCUMENTS, DATA_SOURCES, DATABASE_SCHEMAS, GIT_COMMITS
- document_chunks (content=원문/source of truth, embedding=derived,
  embedding_model/embedding_model_version 컬럼 필수)

USERS, WORKSPACES, TASKS, CHAT_SESSIONS, CHAT_MESSAGES, NOTIFICATIONS,
MESSAGE_CITATIONS, OWNER_CONFIRMATIONS 등 Spring/backend 레포 소유 테이블은
여기에 정의하지 않습니다. 자세한 경계는 .claude/rules/db-boundary.md 참고.

TODO:
- KnowledgeDocument, DataSource, DatabaseSchema, GitCommit 모델 정의
- DocumentChunk 모델 정의 (content, embedding, embedding_model,
  embedding_model_version, document_id FK 등)
- GitCommit.author_id는 인덱싱 시점에 Spring 사용자조회 API로 매핑된 USERS.id 값을
  저장 (이 레포에서 USERS 테이블을 조회/참조/FK 제약하지 않음)
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """AI 도메인 전용 스키마의 ORM Base."""


# TODO: KnowledgeDocument, DataSource, DatabaseSchema, GitCommit, DocumentChunk 모델 정의

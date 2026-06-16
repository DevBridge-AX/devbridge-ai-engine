"""
DB 세션 관리.

config.database_url(AI 도메인 전용 MySQL 스키마, Spring과 별도 DB 계정)을 사용해
SQLAlchemy 엔진/세션을 생성합니다. 동기 세션(pymysql)을 사용하며, 비동기 백그라운드
태스크에서는 SessionLocal()로 독립 세션을 생성하여 사용합니다.
"""

from collections.abc import Generator
from datetime import date

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.database_url,
    pool_pre_ping=True,
    pool_recycle=3600,
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI Depends용 DB 세션 생성자."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def log_embedding_usage(
    db: Session, workspace_id: int, embedding_model: str, tokens: int
) -> None:
    """임베딩 토큰 사용량을 usage_logs에 일 단위로 누적합니다.

    동일 (workspace_id, date, embedding_model) 레코드가 이미 있으면 embedding_tokens를
    더하고, 없으면 새 레코드를 삽입합니다. 동시 쓰기 경합은 Phase 0-1 인덱싱
    특성상 드물어 별도 잠금 없이 처리합니다.
    """
    from app.db.models import UsageLog

    today = date.today()
    existing = db.execute(
        select(UsageLog).where(
            UsageLog.workspace_id == workspace_id,
            UsageLog.date == today,
            UsageLog.embedding_model == embedding_model,
        )
    ).scalar_one_or_none()

    if existing is None:
        db.add(
            UsageLog(
                workspace_id=workspace_id,
                date=today,
                embedding_model=embedding_model,
                embedding_tokens=tokens,
            )
        )
    else:
        existing.embedding_tokens += tokens

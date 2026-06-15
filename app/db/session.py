"""
DB 세션 관리.

config.database_url(AI 도메인 전용 MySQL 스키마, Spring과 별도 DB 계정)을 사용해
SQLAlchemy 엔진/세션을 생성합니다.

TODO:
- engine = create_engine(settings.database_url, ...) 구성
- SessionLocal = sessionmaker(bind=engine, ...) 구성
- FastAPI dependency용 get_db() 구현
"""

from app.config import get_settings

settings = get_settings()

# TODO: engine, SessionLocal 생성


def get_db():
    """FastAPI dependency: DB 세션을 제공. TODO: 구현."""
    raise NotImplementedError

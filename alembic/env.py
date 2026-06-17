from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from app.config import get_settings
from app.db.models import Base

# Alembic Config object
config = context.config

# .env 기반 database_url 동적 주입 (alembic.ini의 sqlalchemy.url 빈 값 덮어쓰기)
config.set_main_option("sqlalchemy.url", get_settings().database_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# autogenerate 대상: AI 도메인 테이블만 (Spring 소유 테이블은 models.py에 정의되지 않음)
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """offline 모드: DB 연결 없이 SQL 스크립트만 생성."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """online 모드: DB에 직접 연결하여 마이그레이션 적용."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

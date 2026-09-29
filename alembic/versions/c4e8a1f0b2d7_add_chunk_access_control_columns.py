"""add access control snapshot columns to document_chunks

Revision ID: c4e8a1f0b2d7
Revises: 7b6c9d1e2f30
Create Date: 2026-09-30

기존 행은 task_id=NULL, sensitivity_level='normal'로 채워져 워크스페이스 전원 조회 가능
(현행 동작과 동일)으로 유지된다. docs/access-control.md Phase 1.
"""

from alembic import op


revision = "c4e8a1f0b2d7"
down_revision = "7b6c9d1e2f30"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE document_chunks
            ADD COLUMN task_id VARCHAR(36) NULL AFTER vector_id,
            ADD COLUMN sensitivity_level VARCHAR(20) NOT NULL DEFAULT 'normal' AFTER task_id,
            ADD INDEX ix_document_chunks_task_id (task_id)
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE document_chunks
            DROP INDEX ix_document_chunks_task_id,
            DROP COLUMN sensitivity_level,
            DROP COLUMN task_id
        """
    )

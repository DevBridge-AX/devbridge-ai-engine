"""add git vector support tables

Revision ID: 7b6c9d1e2f30
Revises: 2a243df8d087
Create Date: 2026-06-22
"""

from alembic import op


revision = "7b6c9d1e2f30"
down_revision = "2a243df8d087"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS document_chunks (
            id BIGINT NOT NULL AUTO_INCREMENT,
            workspace_id VARCHAR(36) NOT NULL,
            source_type VARCHAR(50) NOT NULL,
            source_id VARCHAR(36) NOT NULL,
            content LONGTEXT NOT NULL,
            chunk_metadata JSON NULL,
            embedding_model VARCHAR(100) NOT NULL,
            embedding_model_version VARCHAR(50) NOT NULL,
            vector_id VARCHAR(255) NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            INDEX ix_document_chunks_workspace_id (workspace_id),
            INDEX ix_document_chunks_vector_id (vector_id),
            INDEX ix_document_chunks_source (source_type, source_id)
        )
        """
    )

    op.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_logs (
            id BIGINT NOT NULL AUTO_INCREMENT,
            workspace_id VARCHAR(36) NOT NULL,
            date DATE NOT NULL,
            embedding_model VARCHAR(100) NOT NULL,
            embedding_tokens DECIMAL(12, 1) NOT NULL DEFAULT 0.0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (id),
            UNIQUE KEY uq_usage_logs_workspace_date_model (
                workspace_id,
                date,
                embedding_model
            ),
            INDEX ix_usage_logs_workspace_id (workspace_id)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE IF EXISTS usage_logs")
    op.execute("DROP TABLE IF EXISTS document_chunks")

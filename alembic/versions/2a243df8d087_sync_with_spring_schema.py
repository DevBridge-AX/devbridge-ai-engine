"""sync_with_spring_schema

Revision ID: 2a243df8d087
Revises:
Create Date: 2026-06-21 17:42:08.379363

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2a243df8d087'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema.

    Spring 소유 테이블(data_sources, knowledge_documents, git_commits, database_schemas)은
    이미 Spring ddl-auto:update로 생성되어 있으므로 여기서 CREATE/DROP하지 않습니다.
    AI 전용 테이블(document_chunks, usage_logs)만 생성합니다.
    """
    op.create_table('document_chunks',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('workspace_id', sa.String(length=36), nullable=False),
    sa.Column('source_type', sa.Enum('document', 'git_commit', 'db_schema', name='chunksourcetype'), nullable=False),
    sa.Column('source_id', sa.String(length=36), nullable=False),
    sa.Column('content', sa.Text(), nullable=False),
    sa.Column('chunk_metadata', sa.JSON(), nullable=True),
    sa.Column('embedding_model', sa.String(length=100), nullable=False),
    sa.Column('embedding_model_version', sa.String(length=50), nullable=False),
    sa.Column('vector_id', sa.String(length=255), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_document_chunks_source', 'document_chunks', ['source_type', 'source_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_vector_id'), 'document_chunks', ['vector_id'], unique=False)
    op.create_index(op.f('ix_document_chunks_workspace_id'), 'document_chunks', ['workspace_id'], unique=False)

    op.create_table('usage_logs',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('workspace_id', sa.String(length=36), nullable=False),
    sa.Column('date', sa.Date(), nullable=False),
    sa.Column('embedding_model', sa.String(length=100), nullable=False),
    sa.Column('embedding_tokens', sa.Numeric(precision=12, scale=1), nullable=False),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('workspace_id', 'date', 'embedding_model', name='uq_usage_logs_workspace_date_model')
    )
    op.create_index(op.f('ix_usage_logs_workspace_id'), 'usage_logs', ['workspace_id'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f('ix_usage_logs_workspace_id'), table_name='usage_logs')
    op.drop_table('usage_logs')
    op.drop_index(op.f('ix_document_chunks_workspace_id'), table_name='document_chunks')
    op.drop_index(op.f('ix_document_chunks_vector_id'), table_name='document_chunks')
    op.drop_index('ix_document_chunks_source', table_name='document_chunks')
    op.drop_table('document_chunks')

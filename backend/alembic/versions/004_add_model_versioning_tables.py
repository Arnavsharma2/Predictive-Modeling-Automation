"""Add model versioning tables

Revision ID: 004_model_versioning
Revises: 003_ml_models
Create Date: 2024-01-04 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '004_model_versioning'
down_revision: Union[str, None] = '003_ml_models'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create model_versions table
    op.create_table(
        'model_versions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('performance_metrics', sa.JSON(), nullable=True),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('training_config', sa.JSON(), nullable=True),
        sa.Column('model_path', sa.String(length=500), nullable=True),
        sa.Column('model_size_bytes', sa.Integer(), nullable=True),
        sa.Column('training_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('dataset_size', sa.Integer(), nullable=True),
        sa.Column('training_duration_seconds', sa.Float(), nullable=True),
        sa.Column('feature_importance', sa.JSON(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE')
    )
    op.create_index(op.f('ix_model_versions_id'), 'model_versions', ['id'], unique=False)
    op.create_index(op.f('ix_model_versions_model_id'), 'model_versions', ['model_id'], unique=False)
    op.create_index(op.f('ix_model_versions_version'), 'model_versions', ['version'], unique=False)
    op.create_index(op.f('ix_model_versions_is_active'), 'model_versions', ['is_active'], unique=False)
    # Unique constraint: only one active version per model
    op.create_index('idx_model_versions_model_active', 'model_versions', ['model_id', 'is_active'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_model_versions_model_active', table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_is_active'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_version'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_model_id'), table_name='model_versions')
    op.drop_index(op.f('ix_model_versions_id'), table_name='model_versions')
    op.drop_table('model_versions')


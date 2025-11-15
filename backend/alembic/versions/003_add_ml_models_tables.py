"""Add ML models and training jobs tables

Revision ID: 003_ml_models
Revises: 002_data_ingestion
Create Date: 2024-01-03 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '003_ml_models'
down_revision: Union[str, None] = '002_data_ingestion'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create training_jobs table
    op.create_table(
        'training_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('data_source_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='trainingjobstatus'), nullable=False),
        sa.Column('model_type', sa.String(length=50), nullable=False),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('training_config', sa.JSON(), nullable=True),
        sa.Column('progress', sa.Float(), nullable=True),
        sa.Column('current_epoch', sa.Integer(), nullable=True),
        sa.Column('total_epochs', sa.Integer(), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('metrics', sa.JSON(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='SET NULL')
    )
    op.create_index(op.f('ix_training_jobs_id'), 'training_jobs', ['id'], unique=False)
    op.create_index(op.f('ix_training_jobs_model_id'), 'training_jobs', ['model_id'], unique=False)
    op.create_index(op.f('ix_training_jobs_data_source_id'), 'training_jobs', ['data_source_id'], unique=False)
    op.create_index(op.f('ix_training_jobs_status'), 'training_jobs', ['status'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_training_jobs_status'), table_name='training_jobs')
    op.drop_index(op.f('ix_training_jobs_data_source_id'), table_name='training_jobs')
    op.drop_index(op.f('ix_training_jobs_model_id'), table_name='training_jobs')
    op.drop_index(op.f('ix_training_jobs_id'), table_name='training_jobs')
    op.drop_table('training_jobs')


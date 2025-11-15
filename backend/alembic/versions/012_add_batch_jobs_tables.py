"""Add batch prediction jobs tables

Revision ID: 012_batch_jobs
Revises: 011_data_quality
Create Date: 2025-01-13 14:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '012_batch_jobs'
down_revision: Union[str, None] = '011_data_quality'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create batch_prediction_jobs table
    op.create_table(
        'batch_prediction_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('data_source_id', sa.Integer(), nullable=True),
        sa.Column('status', sa.Enum('PENDING', 'QUEUED', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='batchjobstatus'), nullable=False),
        sa.Column('job_name', sa.String(length=255), nullable=True),
        sa.Column('input_config', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('input_type', sa.String(length=50), nullable=False, server_default='data_source'),
        sa.Column('progress', sa.Float(), nullable=True, server_default='0.0'),
        sa.Column('total_records', sa.Integer(), nullable=True),
        sa.Column('processed_records', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('failed_records', sa.Integer(), nullable=True, server_default='0'),
        sa.Column('result_path', sa.String(length=500), nullable=True),
        sa.Column('result_format', sa.String(length=50), nullable=True, server_default='csv'),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('job_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('scheduled_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('prefect_flow_run_id', sa.String(length=255), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL')
    )
    op.create_index(op.f('ix_batch_prediction_jobs_id'), 'batch_prediction_jobs', ['id'], unique=False)
    op.create_index(op.f('ix_batch_prediction_jobs_model_id'), 'batch_prediction_jobs', ['model_id'], unique=False)
    op.create_index(op.f('ix_batch_prediction_jobs_data_source_id'), 'batch_prediction_jobs', ['data_source_id'], unique=False)
    op.create_index(op.f('ix_batch_prediction_jobs_status'), 'batch_prediction_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_batch_prediction_jobs_prefect_flow_run_id'), 'batch_prediction_jobs', ['prefect_flow_run_id'], unique=False)
    op.create_index(op.f('ix_batch_prediction_jobs_created_by'), 'batch_prediction_jobs', ['created_by'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_batch_prediction_jobs_created_by'), table_name='batch_prediction_jobs')
    op.drop_index(op.f('ix_batch_prediction_jobs_prefect_flow_run_id'), table_name='batch_prediction_jobs')
    op.drop_index(op.f('ix_batch_prediction_jobs_status'), table_name='batch_prediction_jobs')
    op.drop_index(op.f('ix_batch_prediction_jobs_data_source_id'), table_name='batch_prediction_jobs')
    op.drop_index(op.f('ix_batch_prediction_jobs_model_id'), table_name='batch_prediction_jobs')
    op.drop_index(op.f('ix_batch_prediction_jobs_id'), table_name='batch_prediction_jobs')
    op.drop_table('batch_prediction_jobs')
    op.execute("DROP TYPE IF EXISTS batchjobstatus")


"""Add drift reports tables

Revision ID: 010_drift_reports
Revises: 009_add_user_ownership
Create Date: 2025-01-13 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '010_drift_reports'
down_revision: Union[str, None] = '009_add_user_ownership'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create drift_reports table
    op.create_table(
        'drift_reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('drift_type', sa.Enum('DATA_DRIFT', 'CONCEPT_DRIFT', 'PREDICTION_DRIFT', name='drifttype'), nullable=False),
        sa.Column('severity', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'NONE', name='driftseverity'), nullable=False),
        sa.Column('drift_detected', sa.Boolean(), nullable=False),
        sa.Column('drift_results', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('feature_results', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('reference_data_source_id', sa.Integer(), nullable=True),
        sa.Column('reference_samples', sa.Integer(), nullable=True),
        sa.Column('reference_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('current_data_source_id', sa.Integer(), nullable=True),
        sa.Column('current_samples', sa.Integer(), nullable=True),
        sa.Column('current_timestamp', sa.DateTime(timezone=True), nullable=True),
        sa.Column('features_checked', sa.Integer(), nullable=True),
        sa.Column('detection_method', sa.String(length=100), nullable=True),
        sa.Column('threshold_used', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('alert_sent', sa.Boolean(), nullable=True),
        sa.Column('retraining_triggered', sa.Boolean(), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['reference_data_source_id'], ['data_sources.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['current_data_source_id'], ['data_sources.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL')
    )
    op.create_index(op.f('ix_drift_reports_id'), 'drift_reports', ['id'], unique=False)
    op.create_index(op.f('ix_drift_reports_model_id'), 'drift_reports', ['model_id'], unique=False)
    op.create_index(op.f('ix_drift_reports_created_by'), 'drift_reports', ['created_by'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_drift_reports_created_by'), table_name='drift_reports')
    op.drop_index(op.f('ix_drift_reports_model_id'), table_name='drift_reports')
    op.drop_index(op.f('ix_drift_reports_id'), table_name='drift_reports')
    op.drop_table('drift_reports')
    op.execute("DROP TYPE IF EXISTS driftseverity")
    op.execute("DROP TYPE IF EXISTS drifttype")


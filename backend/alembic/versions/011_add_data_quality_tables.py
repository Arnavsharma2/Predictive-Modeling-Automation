"""Add data quality tables

Revision ID: 011_data_quality
Revises: 010_drift_reports
Create Date: 2025-01-13 13:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '011_data_quality'
down_revision: Union[str, None] = '010_drift_reports'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create data_quality_reports table
    op.create_table(
        'data_quality_reports',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('data_source_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('EXCELLENT', 'GOOD', 'FAIR', 'POOR', 'CRITICAL', name='qualitystatus'), nullable=False),
        sa.Column('quality_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('freshness_metrics', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('schema_info', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('sample_size', sa.Integer(), nullable=True),
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('issues', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('recommendations', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL')
    )
    op.create_index(op.f('ix_data_quality_reports_id'), 'data_quality_reports', ['id'], unique=False)
    op.create_index(op.f('ix_data_quality_reports_data_source_id'), 'data_quality_reports', ['data_source_id'], unique=False)
    op.create_index(op.f('ix_data_quality_reports_created_by'), 'data_quality_reports', ['created_by'], unique=False)
    
    # Create data_lineage table
    op.create_table(
        'data_lineage',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_data_source_id', sa.Integer(), nullable=False),
        sa.Column('target_type', sa.String(length=50), nullable=False),
        sa.Column('target_id', sa.Integer(), nullable=False),
        sa.Column('transformation', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('lineage_metadata', postgresql.JSON(astext_type=sa.Text()), nullable=True),
        sa.Column('created_by', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['source_data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['created_by'], ['users.id'], ondelete='SET NULL')
    )
    op.create_index(op.f('ix_data_lineage_id'), 'data_lineage', ['id'], unique=False)
    op.create_index(op.f('ix_data_lineage_source_data_source_id'), 'data_lineage', ['source_data_source_id'], unique=False)
    op.create_index(op.f('ix_data_lineage_target_type'), 'data_lineage', ['target_type'], unique=False)
    op.create_index(op.f('ix_data_lineage_target_id'), 'data_lineage', ['target_id'], unique=False)
    op.create_index(op.f('ix_data_lineage_created_by'), 'data_lineage', ['created_by'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_data_lineage_created_by'), table_name='data_lineage')
    op.drop_index(op.f('ix_data_lineage_target_id'), table_name='data_lineage')
    op.drop_index(op.f('ix_data_lineage_target_type'), table_name='data_lineage')
    op.drop_index(op.f('ix_data_lineage_source_data_source_id'), table_name='data_lineage')
    op.drop_index(op.f('ix_data_lineage_id'), table_name='data_lineage')
    op.drop_table('data_lineage')
    
    op.drop_index(op.f('ix_data_quality_reports_created_by'), table_name='data_quality_reports')
    op.drop_index(op.f('ix_data_quality_reports_data_source_id'), table_name='data_quality_reports')
    op.drop_index(op.f('ix_data_quality_reports_id'), table_name='data_quality_reports')
    op.drop_table('data_quality_reports')
    op.execute("DROP TYPE IF EXISTS qualitystatus")


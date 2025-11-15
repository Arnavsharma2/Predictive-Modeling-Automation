"""Add data ingestion tables

Revision ID: 002_data_ingestion
Revises: 001_initial
Create Date: 2024-01-02 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '002_data_ingestion'
down_revision: Union[str, None] = '001_initial'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create data_points table for time-series data
    op.create_table(
        'data_points',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('data', sa.JSON(), nullable=False),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('timestamp', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['source_id'], ['data_sources.id'], ondelete='CASCADE')
    )
    op.create_index(op.f('ix_data_points_id'), 'data_points', ['id'], unique=False)
    op.create_index(op.f('ix_data_points_source_id'), 'data_points', ['source_id'], unique=False)
    op.create_index(op.f('ix_data_points_timestamp'), 'data_points', ['timestamp'], unique=False)
    op.create_index('idx_data_points_source_timestamp', 'data_points', ['source_id', 'timestamp'], unique=False)
    
    # Convert data_points to TimescaleDB hypertable (if TimescaleDB extension is available)
    # Supabase and some PostgreSQL instances don't have TimescaleDB, so we make this optional
    # Check if TimescaleDB extension exists before creating hypertable
    connection = op.get_bind()
    result = connection.execute(sa.text(
        "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'timescaledb')"
    )).scalar()
    
    if result:
        # TimescaleDB is available, create hypertable
        op.execute("SELECT create_hypertable('data_points', 'timestamp', if_not_exists => TRUE);")
    # If TimescaleDB is not available, the table will work as a regular PostgreSQL table
    
    # Create etl_jobs table
    op.create_table(
        'etl_jobs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('source_id', sa.Integer(), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'RUNNING', 'COMPLETED', 'FAILED', 'CANCELLED', name='etljobstatus'), nullable=False),
        sa.Column('prefect_flow_run_id', sa.String(length=255), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('records_processed', sa.Integer(), nullable=True),
        sa.Column('records_failed', sa.Integer(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['source_id'], ['data_sources.id'], ondelete='CASCADE')
    )
    op.create_index(op.f('ix_etl_jobs_id'), 'etl_jobs', ['id'], unique=False)
    op.create_index(op.f('ix_etl_jobs_source_id'), 'etl_jobs', ['source_id'], unique=False)
    op.create_index(op.f('ix_etl_jobs_status'), 'etl_jobs', ['status'], unique=False)
    op.create_index(op.f('ix_etl_jobs_prefect_flow_run_id'), 'etl_jobs', ['prefect_flow_run_id'], unique=True)


def downgrade() -> None:
    op.drop_index(op.f('ix_etl_jobs_prefect_flow_run_id'), table_name='etl_jobs')
    op.drop_index(op.f('ix_etl_jobs_status'), table_name='etl_jobs')
    op.drop_index(op.f('ix_etl_jobs_source_id'), table_name='etl_jobs')
    op.drop_index(op.f('ix_etl_jobs_id'), table_name='etl_jobs')
    op.drop_table('etl_jobs')
    
    # Drop TimescaleDB hypertable (if it exists)
    op.execute("DROP TABLE IF EXISTS data_points CASCADE;")
    
    # Note: The hypertable drop above will also drop the table and indexes


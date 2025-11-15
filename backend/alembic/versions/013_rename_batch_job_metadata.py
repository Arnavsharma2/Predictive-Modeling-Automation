"""Rename batch_jobs metadata column to job_metadata

Revision ID: 013_rename_batch_job_metadata
Revises: 012_batch_jobs
Create Date: 2025-01-13 15:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision: str = '013_rename_batch_job_metadata'
down_revision: Union[str, None] = '012_batch_jobs'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Check if the table exists and has the metadata column
    conn = op.get_bind()
    inspector = inspect(conn)
    
    # Check if batch_prediction_jobs table exists
    if 'batch_prediction_jobs' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('batch_prediction_jobs')]
        
        # If metadata column exists, rename it to job_metadata
        if 'metadata' in columns and 'job_metadata' not in columns:
            op.alter_column('batch_prediction_jobs', 'metadata', new_column_name='job_metadata')


def downgrade() -> None:
    # Check if the table exists and has the job_metadata column
    conn = op.get_bind()
    inspector = inspect(conn)
    
    # Check if batch_prediction_jobs table exists
    if 'batch_prediction_jobs' in inspector.get_table_names():
        columns = [col['name'] for col in inspector.get_columns('batch_prediction_jobs')]
        
        # If job_metadata column exists, rename it back to metadata
        if 'job_metadata' in columns and 'metadata' not in columns:
            op.alter_column('batch_prediction_jobs', 'job_metadata', new_column_name='metadata')


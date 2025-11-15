"""Initial schema

Revision ID: 001_initial
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(length=255), nullable=False),
        sa.Column('username', sa.String(length=100), nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('is_superuser', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_id'), 'users', ['id'], unique=False)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)
    
    # Create data_sources table
    op.create_table(
        'data_sources',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.Enum('CSV', 'API', 'WEB_SCRAPE', 'CLOUD_STORAGE', name='datasourcetype'), nullable=False),
        sa.Column('status', sa.Enum('ACTIVE', 'INACTIVE', 'ERROR', 'PROCESSING', name='datasourcestatus'), nullable=True),
        sa.Column('config', sa.JSON(), nullable=True),
        sa.Column('description', sa.String(length=1000), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_data_sources_id'), 'data_sources', ['id'], unique=False)
    op.create_index(op.f('ix_data_sources_name'), 'data_sources', ['name'], unique=False)
    
    # Create ml_models table
    op.create_table(
        'ml_models',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('type', sa.Enum('REGRESSION', 'CLASSIFICATION', 'ANOMALY_DETECTION', name='modeltype'), nullable=False),
        sa.Column('status', sa.Enum('TRAINING', 'TRAINED', 'DEPLOYED', 'FAILED', 'ARCHIVED', name='modelstatus'), nullable=True),
        sa.Column('version', sa.String(length=50), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('data_source_id', sa.Integer(), nullable=True),
        sa.Column('features', sa.JSON(), nullable=True),
        sa.Column('hyperparameters', sa.JSON(), nullable=True),
        sa.Column('accuracy', sa.Float(), nullable=True),
        sa.Column('precision', sa.Float(), nullable=True),
        sa.Column('recall', sa.Float(), nullable=True),
        sa.Column('f1_score', sa.Float(), nullable=True),
        sa.Column('rmse', sa.Float(), nullable=True),
        sa.Column('mae', sa.Float(), nullable=True),
        sa.Column('r2_score', sa.Float(), nullable=True),
        sa.Column('model_path', sa.String(length=500), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ml_models_id'), 'ml_models', ['id'], unique=False)
    op.create_index(op.f('ix_ml_models_name'), 'ml_models', ['name'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_ml_models_name'), table_name='ml_models')
    op.drop_index(op.f('ix_ml_models_id'), table_name='ml_models')
    op.drop_table('ml_models')
    op.drop_index(op.f('ix_data_sources_name'), table_name='data_sources')
    op.drop_index(op.f('ix_data_sources_id'), table_name='data_sources')
    op.drop_table('data_sources')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_id'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')


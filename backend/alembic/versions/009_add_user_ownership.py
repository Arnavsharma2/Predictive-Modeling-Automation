"""add user ownership

Revision ID: 009_add_user_ownership
Revises: 008_add_auth_tables
Create Date: 2025-01-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '009_add_user_ownership'
down_revision = '008_add_auth_tables'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add created_by user_id to ml_models table
    op.add_column('ml_models', sa.Column('created_by', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_ml_models_created_by',
        'ml_models', 'users',
        ['created_by'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index(op.f('ix_ml_models_created_by'), 'ml_models', ['created_by'], unique=False)
    
    # Add shared_with JSON field to ml_models for sharing capabilities
    op.add_column('ml_models', sa.Column('shared_with', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Add created_by user_id to data_sources table
    op.add_column('data_sources', sa.Column('created_by', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_data_sources_created_by',
        'data_sources', 'users',
        ['created_by'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index(op.f('ix_data_sources_created_by'), 'data_sources', ['created_by'], unique=False)
    
    # Add shared_with JSON field to data_sources for sharing capabilities
    op.add_column('data_sources', sa.Column('shared_with', postgresql.JSON(astext_type=sa.Text()), nullable=True))
    
    # Add created_by user_id to training_jobs table
    op.add_column('training_jobs', sa.Column('created_by', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'fk_training_jobs_created_by',
        'training_jobs', 'users',
        ['created_by'], ['id'],
        ondelete='SET NULL'
    )
    op.create_index(op.f('ix_training_jobs_created_by'), 'training_jobs', ['created_by'], unique=False)


def downgrade() -> None:
    # Remove training_jobs created_by
    op.drop_index(op.f('ix_training_jobs_created_by'), table_name='training_jobs')
    op.drop_constraint('fk_training_jobs_created_by', 'training_jobs', type_='foreignkey')
    op.drop_column('training_jobs', 'created_by')
    
    # Remove data_sources shared_with and created_by
    op.drop_column('data_sources', 'shared_with')
    op.drop_index(op.f('ix_data_sources_created_by'), table_name='data_sources')
    op.drop_constraint('fk_data_sources_created_by', 'data_sources', type_='foreignkey')
    op.drop_column('data_sources', 'created_by')
    
    # Remove ml_models shared_with and created_by
    op.drop_column('ml_models', 'shared_with')
    op.drop_index(op.f('ix_ml_models_created_by'), table_name='ml_models')
    op.drop_constraint('fk_ml_models_created_by', 'ml_models', type_='foreignkey')
    op.drop_column('ml_models', 'created_by')


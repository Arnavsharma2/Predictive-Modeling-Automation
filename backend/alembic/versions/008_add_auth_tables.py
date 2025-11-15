"""add auth tables

Revision ID: 008_add_auth_tables
Revises: 007_original_columns
Create Date: 2025-01-13 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '008_add_auth_tables'
down_revision = '007_original_columns'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create enum type for user roles (only if it doesn't exist)
    op.execute("DO $$ BEGIN CREATE TYPE userrole AS ENUM ('admin', 'data_scientist', 'viewer'); EXCEPTION WHEN duplicate_object THEN null; END $$;")

    # Add role column to users table with proper type casting
    op.execute("ALTER TABLE users ADD COLUMN role userrole DEFAULT 'viewer'::userrole NOT NULL")

    # Create API keys table
    op.create_table(
        'api_keys',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('key_prefix', sa.String(length=10), nullable=False),
        sa.Column('hashed_key', sa.String(length=255), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True, server_default='true'),
        sa.Column('last_used_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('expires_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_api_keys_key_prefix'), 'api_keys', ['key_prefix'], unique=False)


def downgrade() -> None:
    # Drop API keys table
    op.drop_index(op.f('ix_api_keys_key_prefix'), table_name='api_keys')
    op.drop_table('api_keys')

    # Remove role column from users table
    op.drop_column('users', 'role')

    # Drop enum type
    op.execute('DROP TYPE userrole')

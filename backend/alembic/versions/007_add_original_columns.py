"""Add original_columns to ml_models table

Revision ID: 007_original_columns
Revises: 006_feature_name_mapping
Create Date: 2024-01-06 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '007_original_columns'
down_revision: Union[str, None] = '006_feature_name_mapping'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add original_columns column to ml_models table
    op.add_column(
        'ml_models',
        sa.Column('original_columns', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    # Remove original_columns column from ml_models table
    op.drop_column('ml_models', 'original_columns')

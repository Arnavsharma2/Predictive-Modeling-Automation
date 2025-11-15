"""Add feature_name_mapping to ml_models table

Revision ID: 006_feature_name_mapping
Revises: 005_alerts_ab_tests
Create Date: 2024-01-05 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '006_feature_name_mapping'
down_revision: Union[str, None] = '005_alerts_ab_tests'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add feature_name_mapping column to ml_models table
    op.add_column(
        'ml_models',
        sa.Column('feature_name_mapping', postgresql.JSON(astext_type=sa.Text()), nullable=True)
    )


def downgrade() -> None:
    # Remove feature_name_mapping column from ml_models table
    op.drop_column('ml_models', 'feature_name_mapping')


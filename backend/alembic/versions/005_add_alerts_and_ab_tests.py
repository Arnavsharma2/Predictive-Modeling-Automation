"""Add alerts and A/B testing tables

Revision ID: 005_alerts_ab_tests
Revises: 004_add_model_versioning_tables
Create Date: 2024-01-01 12:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '005_alerts_ab_tests'
down_revision: Union[str, None] = '004_model_versioning'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create alert_configs table
    op.create_table(
        'alert_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('alert_type', sa.Enum('ANOMALY_DETECTION', 'MODEL_PERFORMANCE', 'PIPELINE_FAILURE', 'DATA_QUALITY', 'SYSTEM', name='alerttype'), nullable=False),
        sa.Column('severity', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='alertseverity'), nullable=False),
        sa.Column('enabled', sa.Boolean(), nullable=False),
        sa.Column('conditions', sa.JSON(), nullable=False),
        sa.Column('notification_channels', sa.JSON(), nullable=False),
        sa.Column('email_recipients', sa.JSON(), nullable=True),
        sa.Column('webhook_url', sa.String(length=500), nullable=True),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('data_source_id', sa.Integer(), nullable=True),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('name')
    )
    op.create_index(op.f('ix_alert_configs_id'), 'alert_configs', ['id'], unique=False)
    op.create_index(op.f('ix_alert_configs_name'), 'alert_configs', ['name'], unique=True)
    op.create_index(op.f('ix_alert_configs_alert_type'), 'alert_configs', ['alert_type'], unique=False)
    op.create_index(op.f('ix_alert_configs_enabled'), 'alert_configs', ['enabled'], unique=False)
    op.create_index(op.f('ix_alert_configs_model_id'), 'alert_configs', ['model_id'], unique=False)
    
    # Create alerts table
    op.create_table(
        'alerts',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('config_id', sa.Integer(), nullable=False),
        sa.Column('alert_type', sa.Enum('ANOMALY_DETECTION', 'MODEL_PERFORMANCE', 'PIPELINE_FAILURE', 'DATA_QUALITY', 'SYSTEM', name='alerttype'), nullable=False),
        sa.Column('severity', sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='alertseverity'), nullable=False),
        sa.Column('status', sa.Enum('PENDING', 'SENT', 'ACKNOWLEDGED', 'RESOLVED', 'FAILED', name='alertstatus'), nullable=False),
        sa.Column('title', sa.String(length=500), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('details', sa.JSON(), nullable=True),
        sa.Column('model_id', sa.Integer(), nullable=True),
        sa.Column('data_source_id', sa.Integer(), nullable=True),
        sa.Column('etl_job_id', sa.Integer(), nullable=True),
        sa.Column('notifications_sent', sa.JSON(), nullable=True),
        sa.Column('acknowledged_by', sa.String(length=200), nullable=True),
        sa.Column('acknowledged_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('resolved_by', sa.String(length=200), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['config_id'], ['alert_configs.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['data_source_id'], ['data_sources.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['etl_job_id'], ['etl_jobs.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_alerts_id'), 'alerts', ['id'], unique=False)
    op.create_index(op.f('ix_alerts_config_id'), 'alerts', ['config_id'], unique=False)
    op.create_index(op.f('ix_alerts_alert_type'), 'alerts', ['alert_type'], unique=False)
    op.create_index(op.f('ix_alerts_severity'), 'alerts', ['severity'], unique=False)
    op.create_index(op.f('ix_alerts_status'), 'alerts', ['status'], unique=False)
    op.create_index(op.f('ix_alerts_model_id'), 'alerts', ['model_id'], unique=False)
    op.create_index(op.f('ix_alerts_created_at'), 'alerts', ['created_at'], unique=False)
    
    # Create ab_tests table
    op.create_table(
        'ab_tests',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.Enum('DRAFT', 'RUNNING', 'PAUSED', 'COMPLETED', 'CANCELLED', name='abteststatus'), nullable=False),
        sa.Column('model_id', sa.Integer(), nullable=False),
        sa.Column('control_version_id', sa.Integer(), nullable=True),
        sa.Column('treatment_version_id', sa.Integer(), nullable=True),
        sa.Column('control_traffic_percentage', sa.Float(), nullable=False),
        sa.Column('treatment_traffic_percentage', sa.Float(), nullable=False),
        sa.Column('routing_strategy', sa.String(length=50), nullable=False),
        sa.Column('start_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('end_date', sa.DateTime(timezone=True), nullable=True),
        sa.Column('min_samples', sa.Integer(), nullable=False),
        sa.Column('control_metrics', sa.JSON(), nullable=True),
        sa.Column('treatment_metrics', sa.JSON(), nullable=True),
        sa.Column('statistical_significance', sa.Float(), nullable=True),
        sa.Column('winner', sa.String(length=20), nullable=True),
        sa.Column('metadata', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['model_id'], ['ml_models.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['control_version_id'], ['model_versions.id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['treatment_version_id'], ['model_versions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ab_tests_id'), 'ab_tests', ['id'], unique=False)
    op.create_index(op.f('ix_ab_tests_name'), 'ab_tests', ['name'], unique=False)
    op.create_index(op.f('ix_ab_tests_status'), 'ab_tests', ['status'], unique=False)
    op.create_index(op.f('ix_ab_tests_model_id'), 'ab_tests', ['model_id'], unique=False)
    
    # Create ab_test_predictions table
    op.create_table(
        'ab_test_predictions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('test_id', sa.Integer(), nullable=False),
        sa.Column('version_id', sa.Integer(), nullable=False),
        sa.Column('variant', sa.String(length=20), nullable=False),
        sa.Column('input_data', sa.JSON(), nullable=False),
        sa.Column('prediction', sa.JSON(), nullable=True),
        sa.Column('actual_value', sa.JSON(), nullable=True),
        sa.Column('user_id', sa.String(length=200), nullable=True),
        sa.Column('session_id', sa.String(length=200), nullable=True),
        sa.Column('latency_ms', sa.Float(), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['test_id'], ['ab_tests.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['version_id'], ['model_versions.id'], ondelete='SET NULL'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_ab_test_predictions_id'), 'ab_test_predictions', ['id'], unique=False)
    op.create_index(op.f('ix_ab_test_predictions_test_id'), 'ab_test_predictions', ['test_id'], unique=False)
    op.create_index(op.f('ix_ab_test_predictions_version_id'), 'ab_test_predictions', ['version_id'], unique=False)
    op.create_index(op.f('ix_ab_test_predictions_variant'), 'ab_test_predictions', ['variant'], unique=False)
    op.create_index(op.f('ix_ab_test_predictions_user_id'), 'ab_test_predictions', ['user_id'], unique=False)
    op.create_index(op.f('ix_ab_test_predictions_session_id'), 'ab_test_predictions', ['session_id'], unique=False)
    op.create_index(op.f('ix_ab_test_predictions_created_at'), 'ab_test_predictions', ['created_at'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_ab_test_predictions_created_at'), table_name='ab_test_predictions')
    op.drop_index(op.f('ix_ab_test_predictions_session_id'), table_name='ab_test_predictions')
    op.drop_index(op.f('ix_ab_test_predictions_user_id'), table_name='ab_test_predictions')
    op.drop_index(op.f('ix_ab_test_predictions_variant'), table_name='ab_test_predictions')
    op.drop_index(op.f('ix_ab_test_predictions_version_id'), table_name='ab_test_predictions')
    op.drop_index(op.f('ix_ab_test_predictions_test_id'), table_name='ab_test_predictions')
    op.drop_index(op.f('ix_ab_test_predictions_id'), table_name='ab_test_predictions')
    op.drop_table('ab_test_predictions')
    op.drop_index(op.f('ix_ab_tests_model_id'), table_name='ab_tests')
    op.drop_index(op.f('ix_ab_tests_status'), table_name='ab_tests')
    op.drop_index(op.f('ix_ab_tests_name'), table_name='ab_tests')
    op.drop_index(op.f('ix_ab_tests_id'), table_name='ab_tests')
    op.drop_table('ab_tests')
    op.drop_index(op.f('ix_alerts_created_at'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_model_id'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_status'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_severity'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_alert_type'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_config_id'), table_name='alerts')
    op.drop_index(op.f('ix_alerts_id'), table_name='alerts')
    op.drop_table('alerts')
    op.drop_index(op.f('ix_alert_configs_model_id'), table_name='alert_configs')
    op.drop_index(op.f('ix_alert_configs_enabled'), table_name='alert_configs')
    op.drop_index(op.f('ix_alert_configs_alert_type'), table_name='alert_configs')
    op.drop_index(op.f('ix_alert_configs_name'), table_name='alert_configs')
    op.drop_index(op.f('ix_alert_configs_id'), table_name='alert_configs')
    op.drop_table('alert_configs')
    
    # Drop enums
    sa.Enum(name='abteststatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='alertstatus').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='alertseverity').drop(op.get_bind(), checkfirst=True)
    sa.Enum(name='alerttype').drop(op.get_bind(), checkfirst=True)


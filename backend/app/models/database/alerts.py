"""
Alert database models.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum, Boolean
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class AlertType(str, enum.Enum):
    """Alert types."""
    ANOMALY_DETECTION = "anomaly_detection"
    MODEL_PERFORMANCE = "model_performance"
    PIPELINE_FAILURE = "pipeline_failure"
    DATA_QUALITY = "data_quality"
    SYSTEM = "system"


class AlertSeverity(str, enum.Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, enum.Enum):
    """Alert status."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"


class AlertConfig(Base):
    """Alert configuration."""
    __tablename__ = "alert_configs"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, unique=True, index=True)
    alert_type = Column(SQLEnum(AlertType), nullable=False, index=True)
    severity = Column(SQLEnum(AlertSeverity), default=AlertSeverity.MEDIUM, nullable=False)
    enabled = Column(Boolean, default=True, nullable=False, index=True)
    
    # Conditions
    conditions = Column(JSON, nullable=False)  # e.g., {"threshold": 0.05, "metric": "accuracy"}
    
    # Notification channels
    notification_channels = Column(JSON, nullable=False)  # e.g., ["email", "webhook"]
    email_recipients = Column(JSON, nullable=True)  # List of email addresses
    webhook_url = Column(String(500), nullable=True)
    
    # Model/entity association
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=True, index=True)
    
    # Metadata
    description = Column(Text, nullable=True)
    meta_data = Column("metadata", JSON, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class Alert(Base):
    """Alert instance."""
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    config_id = Column(Integer, ForeignKey("alert_configs.id", ondelete="CASCADE"), nullable=False, index=True)
    alert_type = Column(SQLEnum(AlertType), nullable=False, index=True)
    severity = Column(SQLEnum(AlertSeverity), nullable=False, index=True)
    status = Column(SQLEnum(AlertStatus), default=AlertStatus.PENDING, nullable=False, index=True)
    
    # Alert details
    title = Column(String(500), nullable=False)
    message = Column(Text, nullable=False)
    details = Column(JSON, nullable=True)  # Additional context
    
    # Associated entities
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="SET NULL"), nullable=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True, index=True)
    etl_job_id = Column(Integer, ForeignKey("etl_jobs.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Notification tracking
    notifications_sent = Column(JSON, nullable=True)  # Track which channels were notified
    acknowledged_by = Column(String(200), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_by = Column(String(200), nullable=True)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


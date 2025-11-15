"""
Drift detection database models.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class DriftType(str, enum.Enum):
    """Drift type enumeration."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"


class DriftSeverity(str, enum.Enum):
    """Drift severity enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NONE = "none"


class DriftReport(Base):
    """Drift detection report model."""
    
    __tablename__ = "drift_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    drift_type = Column(SQLEnum(DriftType), nullable=False)
    severity = Column(SQLEnum(DriftSeverity), nullable=False, default=DriftSeverity.NONE)
    
    # Drift detection results
    drift_detected = Column(Boolean, nullable=False, default=False)
    drift_results = Column(JSON, nullable=True)  # Full drift detection results
    feature_results = Column(JSON, nullable=True)  # Per-feature drift results
    
    # Reference data info
    reference_data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    reference_samples = Column(Integer, nullable=True)
    reference_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Current data info
    current_data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True)
    current_samples = Column(Integer, nullable=True)
    current_timestamp = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    features_checked = Column(Integer, nullable=True)
    detection_method = Column(String(100), nullable=True)  # e.g., "PSI", "KS", "combined"
    threshold_used = Column(JSON, nullable=True)  # Thresholds used for detection
    
    # Actions taken
    alert_sent = Column(Boolean, default=False)
    retraining_triggered = Column(Boolean, default=False)
    notes = Column(Text, nullable=True)
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


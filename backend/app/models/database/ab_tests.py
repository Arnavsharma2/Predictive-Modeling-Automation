"""
A/B testing database models.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum, Boolean, Float
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class ABTestStatus(str, enum.Enum):
    """A/B test status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class ABTest(Base):
    """A/B test configuration."""
    __tablename__ = "ab_tests"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False, index=True)
    description = Column(Text, nullable=True)
    status = Column(SQLEnum(ABTestStatus), default=ABTestStatus.DRAFT, nullable=False, index=True)
    
    # Model assignments
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    control_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    treatment_version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=True)
    
    # Traffic splitting
    control_traffic_percentage = Column(Float, default=50.0, nullable=False)  # 0-100
    treatment_traffic_percentage = Column(Float, default=50.0, nullable=False)  # 0-100
    
    # Routing strategy
    routing_strategy = Column(String(50), default="random", nullable=False)  # random, user_id, session_id
    
    # Test configuration
    start_date = Column(DateTime(timezone=True), nullable=True)
    end_date = Column(DateTime(timezone=True), nullable=True)
    min_samples = Column(Integer, default=1000, nullable=False)  # Minimum samples for statistical significance
    
    # Results
    control_metrics = Column(JSON, nullable=True)  # Performance metrics for control
    treatment_metrics = Column(JSON, nullable=True)  # Performance metrics for treatment
    statistical_significance = Column(Float, nullable=True)  # p-value
    winner = Column(String(20), nullable=True)  # "control", "treatment", or "none"
    
    # Metadata
    meta_data = Column("metadata", JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class ABTestPrediction(Base):
    """Track predictions made during A/B test."""
    __tablename__ = "ab_test_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    test_id = Column(Integer, ForeignKey("ab_tests.id", ondelete="CASCADE"), nullable=False, index=True)
    version_id = Column(Integer, ForeignKey("model_versions.id", ondelete="SET NULL"), nullable=False, index=True)
    variant = Column(String(20), nullable=False, index=True)  # "control" or "treatment"
    
    # Prediction details
    input_data = Column(JSON, nullable=False)
    prediction = Column(JSON, nullable=True)
    actual_value = Column(JSON, nullable=True)  # Ground truth (if available)
    
    # Routing info
    user_id = Column(String(200), nullable=True, index=True)
    session_id = Column(String(200), nullable=True, index=True)
    
    # Performance tracking
    latency_ms = Column(Float, nullable=True)
    error = Column(Text, nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now(), index=True)


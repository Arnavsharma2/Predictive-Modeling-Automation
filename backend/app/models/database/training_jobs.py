"""
Training job tracking database model.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum, Float
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class TrainingJobStatus(str, enum.Enum):
    """Training job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingJob(Base):
    """Training job tracking model."""
    
    __tablename__ = "training_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=True, index=True)  # Nullable for initial training
    data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True, index=True)
    status = Column(SQLEnum(TrainingJobStatus), default=TrainingJobStatus.PENDING, nullable=False, index=True)
    
    # Training configuration
    model_type = Column(String(50), nullable=False)  # regression, classification, anomaly_detection
    hyperparameters = Column(JSON, nullable=True)  # Training hyperparameters
    training_config = Column(JSON, nullable=True)  # Additional training configuration
    
    # Progress tracking
    progress = Column(Float, default=0.0)  # Progress percentage (0-100)
    current_epoch = Column(Integer, default=0)  # Current epoch/iteration
    total_epochs = Column(Integer, nullable=True)  # Total epochs/iterations
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    error_message = Column(Text, nullable=True)
    metrics = Column(JSON, nullable=True)  # Training metrics (loss, accuracy, etc.)
    
    # Metadata
    meta_data = Column("metadata", JSON, nullable=True)  # Additional job metadata
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


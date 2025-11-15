"""
Model versioning database model.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, Boolean, ForeignKey, Text
from sqlalchemy.sql import func
from app.core.database import Base


class ModelVersion(Base):
    """Model version tracking model."""
    
    __tablename__ = "model_versions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    version = Column(String(50), nullable=False, index=True)  # e.g., "1.0.0", "1.1.0"
    
    # Performance metrics (stored as JSON for flexibility)
    performance_metrics = Column(JSON, nullable=True)  # All metrics (accuracy, precision, recall, etc.)
    
    # Model configuration
    features = Column(JSON, nullable=True)  # List of feature names used
    hyperparameters = Column(JSON, nullable=True)  # Model hyperparameters
    training_config = Column(JSON, nullable=True)  # Training configuration used
    
    # Model artifacts
    model_path = Column(String(500), nullable=True)  # Path to model artifact in cloud storage
    model_size_bytes = Column(Integer, nullable=True)  # Model file size in bytes
    
    # Training metadata
    training_date = Column(DateTime(timezone=True), nullable=True)
    dataset_size = Column(Integer, nullable=True)  # Number of samples used for training
    training_duration_seconds = Column(Float, nullable=True)  # Training time in seconds
    
    # Feature importance
    feature_importance = Column(JSON, nullable=True)  # Feature importance scores
    
    # Version status
    is_active = Column(Boolean, default=False, index=True)  # Only one active version per model
    is_archived = Column(Boolean, default=False)
    
    # Metadata
    notes = Column(Text, nullable=True)  # Version notes/description
    tags = Column(JSON, nullable=True)  # List of tags for filtering/searching
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


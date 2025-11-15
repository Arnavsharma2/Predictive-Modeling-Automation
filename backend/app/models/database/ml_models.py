"""
ML model database model.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, Boolean, Text, Enum as SQLEnum, ForeignKey
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class ModelType(str, enum.Enum):
    """ML model type enumeration."""
    REGRESSION = "regression"
    CLASSIFICATION = "classification"
    ANOMALY_DETECTION = "anomaly_detection"


class ModelStatus(str, enum.Enum):
    """Model status enumeration."""
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ARCHIVED = "archived"


class MLModel(Base):
    """ML model metadata model."""
    
    __tablename__ = "ml_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    type = Column(SQLEnum(ModelType), nullable=False)
    status = Column(SQLEnum(ModelStatus), default=ModelStatus.TRAINING)
    version = Column(String(50), nullable=False, default="1.0.0")
    description = Column(Text, nullable=True)
    
    # Model metadata
    data_source_id = Column(Integer, nullable=True)  # Reference to data source used for training
    features = Column(JSON, nullable=True)  # List of feature names (after preprocessing/transformation)
    original_columns = Column(JSON, nullable=True)  # List of original column names (before preprocessing)
    feature_name_mapping = Column(JSON, nullable=True)  # Mapping from technical feature names to readable names
    hyperparameters = Column(JSON, nullable=True)  # Model hyperparameters
    feature_importance = Column(JSON, nullable=True)  # Feature importance scores
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    rmse = Column(Float, nullable=True)  # For regression
    mae = Column(Float, nullable=True)  # For regression
    r2_score = Column(Float, nullable=True)  # For regression
    
    # Storage
    model_path = Column(String(500), nullable=True)  # Path to model artifact in cloud storage
    is_active = Column(Boolean, default=True)
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    shared_with = Column(JSON, nullable=True)  # JSON array of user IDs or team IDs that have access
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


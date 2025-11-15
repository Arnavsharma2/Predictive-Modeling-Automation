"""
Batch prediction job tracking database model.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum, Float
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class BatchJobStatus(str, enum.Enum):
    """Batch prediction job status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class BatchPredictionJob(Base):
    """Batch prediction job tracking model."""
    
    __tablename__ = "batch_prediction_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("ml_models.id", ondelete="CASCADE"), nullable=False, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True, index=True)
    
    # Job configuration
    status = Column(SQLEnum(BatchJobStatus), default=BatchJobStatus.PENDING, nullable=False, index=True)
    job_name = Column(String(255), nullable=True)  # Optional name for the job
    
    # Input configuration
    input_config = Column(JSON, nullable=True)  # Configuration for input data (file path, query, etc.)
    input_type = Column(String(50), nullable=False, default="data_source")  # data_source, file, query
    
    # Progress tracking
    progress = Column(Float, default=0.0)  # Progress percentage (0-100)
    total_records = Column(Integer, nullable=True)  # Total records to process
    processed_records = Column(Integer, default=0)  # Records processed so far
    failed_records = Column(Integer, default=0)  # Records that failed
    
    # Results
    result_path = Column(String(500), nullable=True)  # Path to results file (S3, local, etc.)
    result_format = Column(String(50), default="csv")  # csv, json, parquet
    error_message = Column(Text, nullable=True)
    job_metadata = Column(JSON, nullable=True)  # Additional job metadata
    
    # Scheduling
    scheduled_at = Column(DateTime(timezone=True), nullable=True)  # When to run (for scheduled jobs)
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Prefect integration
    prefect_flow_run_id = Column(String(255), nullable=True, index=True)  # Prefect flow run ID
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


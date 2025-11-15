"""
ETL job tracking database model.
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class ETLJobStatus(str, enum.Enum):
    """ETL job status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ETLJob(Base):
    """ETL job tracking model."""
    
    __tablename__ = "etl_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(SQLEnum(ETLJobStatus), default=ETLJobStatus.PENDING, nullable=False, index=True)
    prefect_flow_run_id = Column(String(255), nullable=True, unique=True, index=True)  # Prefect flow run ID
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_message = Column(Text, nullable=True)
    records_processed = Column(Integer, default=0)  # Number of records processed
    records_failed = Column(Integer, default=0)  # Number of records that failed
    meta_data = Column("metadata", JSON, nullable=True)  # Additional job metadata (e.g., config, stats)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


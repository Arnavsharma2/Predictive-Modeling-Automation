"""
Data lineage database model.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from app.core.database import Base


class DataLineage(Base):
    """Data lineage tracking model."""
    
    __tablename__ = "data_lineage"
    
    id = Column(Integer, primary_key=True, index=True)
    source_data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    
    # Target information
    target_type = Column(String(50), nullable=False, index=True)  # "data_source", "model", "training_job"
    target_id = Column(Integer, nullable=False, index=True)
    
    # Transformation details
    transformation = Column(JSON, nullable=True)  # Details about the transformation applied
    
    # Metadata
    description = Column(Text, nullable=True)
    lineage_metadata = Column(JSON, nullable=True)
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


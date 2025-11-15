"""
Data points database model for time-series data.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, ForeignKey, Index
from sqlalchemy.sql import func
from app.core.database import Base


class DataPoint(Base):
    """Data point model for storing time-series data."""
    
    __tablename__ = "data_points"
    
    id = Column(Integer, primary_key=True, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    data = Column(JSON, nullable=False)  # Store the actual data point as JSON
    meta_data = Column("metadata", JSON, nullable=True)  # Additional metadata (e.g., quality scores, tags)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Index for time-series queries
    __table_args__ = (
        Index('idx_data_points_source_timestamp', 'source_id', 'timestamp'),
    )


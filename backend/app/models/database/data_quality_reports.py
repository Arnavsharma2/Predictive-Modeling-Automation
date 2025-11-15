"""
Data quality database models.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, Float, Boolean, ForeignKey, Text, Enum as SQLEnum
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class QualityStatus(str, enum.Enum):
    """Data quality status enumeration."""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class DataQualityReport(Base):
    """Data quality report model."""
    
    __tablename__ = "data_quality_reports"
    
    id = Column(Integer, primary_key=True, index=True)
    data_source_id = Column(Integer, ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, index=True)
    status = Column(SQLEnum(QualityStatus), nullable=False, default=QualityStatus.FAIR)
    
    # Quality metrics
    quality_metrics = Column(JSON, nullable=True)  # Completeness, uniqueness, validity, consistency
    freshness_metrics = Column(JSON, nullable=True)  # Data freshness information
    schema_info = Column(JSON, nullable=True)  # Schema information
    
    # Summary
    sample_size = Column(Integer, nullable=True)
    overall_score = Column(Float, nullable=True)  # Overall quality score (0-1)
    
    # Issues detected
    issues = Column(JSON, nullable=True)  # List of detected issues
    recommendations = Column(JSON, nullable=True)  # Recommendations for improvement
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


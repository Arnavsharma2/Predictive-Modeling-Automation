"""
Pydantic schemas for data quality.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from app.models.database.data_quality_reports import QualityStatus


class DataQualityCheckRequest(BaseModel):
    """Schema for data quality check request."""
    model_config = ConfigDict(protected_namespaces=())
    
    data_source_id: int = Field(..., description="ID of the data source to check")


class DataQualityReportResponse(BaseModel):
    """Schema for data quality report response."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    data_source_id: int
    status: str
    quality_metrics: Optional[Dict[str, Any]] = None
    freshness_metrics: Optional[Dict[str, Any]] = None
    schema_info: Optional[Dict[str, Any]] = None
    sample_size: Optional[int] = None
    overall_score: Optional[float] = None
    issues: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_orm(cls, report):
        """Create from ORM model."""
        return cls(
            id=report.id,
            data_source_id=report.data_source_id,
            status=report.status.value,
            quality_metrics=report.quality_metrics,
            freshness_metrics=report.freshness_metrics,
            schema_info=report.schema_info,
            sample_size=report.sample_size,
            overall_score=report.overall_score,
            issues=report.issues,
            recommendations=report.recommendations,
            created_at=report.created_at,
            updated_at=report.updated_at
        )


class DataQualityReportListResponse(BaseModel):
    """Schema for data quality report list response."""
    model_config = ConfigDict(protected_namespaces=())
    
    reports: List[DataQualityReportResponse]
    total: int
    limit: int
    offset: int


class DataProfileResponse(BaseModel):
    """Schema for data profile response."""
    model_config = ConfigDict(protected_namespaces=())
    
    overview: Dict[str, Any]
    columns: Dict[str, Any]
    correlations: Dict[str, Any]
    missing_data: Dict[str, Any]


class DataLineageResponse(BaseModel):
    """Schema for data lineage response."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    source_data_source_id: int
    target_type: str
    target_id: int
    transformation: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    created_at: datetime
    
    @classmethod
    def from_orm(cls, lineage):
        """Create from ORM model."""
        return cls(
            id=lineage.id,
            source_data_source_id=lineage.source_data_source_id,
            target_type=lineage.target_type,
            target_id=lineage.target_id,
            transformation=lineage.transformation,
            description=lineage.description,
            created_at=lineage.created_at
        )


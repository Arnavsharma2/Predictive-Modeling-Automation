"""
Pydantic schemas for drift detection.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from app.models.database.drift_reports import DriftType, DriftSeverity


class DriftCheckRequest(BaseModel):
    """Schema for drift check request."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_id: int = Field(..., description="ID of the model to check")
    current_data_source_id: Optional[int] = Field(None, description="ID of current data source")
    features: Optional[List[str]] = Field(None, description="Features to check (None = all model features)")


class FeatureDriftResult(BaseModel):
    """Schema for feature-level drift results."""
    model_config = ConfigDict(protected_namespaces=())
    
    feature: str
    drift_detected: bool
    severity: Optional[str] = None
    drift_reasons: Optional[List[str]] = None
    metrics: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None


class DriftReportResponse(BaseModel):
    """Schema for drift report response."""
    model_config = ConfigDict(protected_namespaces=())
    
    id: int
    model_id: int
    drift_type: str
    severity: str
    drift_detected: bool
    drift_results: Optional[Dict[str, Any]] = None
    feature_results: Optional[Dict[str, Any]] = None
    reference_samples: Optional[int] = None
    current_samples: Optional[int] = None
    features_checked: Optional[int] = None
    detection_method: Optional[str] = None
    threshold_used: Optional[Dict[str, Any]] = None
    alert_sent: bool = False
    retraining_triggered: bool = False
    notes: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    @classmethod
    def from_orm(cls, drift_report):
        """Create from ORM model."""
        return cls(
            id=drift_report.id,
            model_id=drift_report.model_id,
            drift_type=drift_report.drift_type.value,
            severity=drift_report.severity.value,
            drift_detected=drift_report.drift_detected,
            drift_results=drift_report.drift_results,
            feature_results=drift_report.feature_results,
            reference_samples=drift_report.reference_samples,
            current_samples=drift_report.current_samples,
            features_checked=drift_report.features_checked,
            detection_method=drift_report.detection_method,
            threshold_used=drift_report.threshold_used,
            alert_sent=drift_report.alert_sent,
            retraining_triggered=drift_report.retraining_triggered,
            notes=drift_report.notes,
            created_at=drift_report.created_at,
            updated_at=drift_report.updated_at
        )


class DriftReportListResponse(BaseModel):
    """Schema for drift report list response."""
    model_config = ConfigDict(protected_namespaces=())
    
    reports: List[DriftReportResponse]
    total: int
    limit: int
    offset: int


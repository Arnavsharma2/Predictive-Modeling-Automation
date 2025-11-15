"""
Alert schemas.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.database.alerts import AlertType, AlertSeverity, AlertStatus


class AlertConfigCreate(BaseModel):
    """Schema for creating alert configuration."""
    name: str = Field(..., min_length=1, max_length=200)
    alert_type: str = Field(..., description="Type of alert")
    conditions: Dict[str, Any] = Field(..., description="Alert conditions")
    notification_channels: List[str] = Field(..., description="Notification channels")
    severity: Optional[str] = Field("medium", description="Alert severity")
    model_id: Optional[int] = None
    data_source_id: Optional[int] = None
    email_recipients: Optional[List[str]] = None
    webhook_url: Optional[str] = None
    description: Optional[str] = None


class AlertConfigResponse(BaseModel):
    """Schema for alert configuration response."""
    id: int
    name: str
    alert_type: str
    severity: str
    enabled: bool
    conditions: Dict[str, Any]
    notification_channels: List[str]
    email_recipients: Optional[List[str]]
    webhook_url: Optional[str]
    model_id: Optional[int]
    data_source_id: Optional[int]
    description: Optional[str]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AlertResponse(BaseModel):
    """Schema for alert response."""
    id: int
    config_id: int
    alert_type: str
    severity: str
    status: str
    title: str
    message: str
    details: Optional[Dict[str, Any]]
    model_id: Optional[int]
    data_source_id: Optional[int]
    etl_job_id: Optional[int]
    notifications_sent: Optional[List[str]]
    acknowledged_by: Optional[str]
    acknowledged_at: Optional[datetime]
    resolved_by: Optional[str]
    resolved_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class AlertListResponse(BaseModel):
    """Schema for alert list response."""
    alerts: List[AlertResponse]
    total: int


class AlertAcknowledgeRequest(BaseModel):
    """Schema for acknowledging alert."""
    acknowledged_by: str = Field(..., min_length=1)


class AlertResolveRequest(BaseModel):
    """Schema for resolving alert."""
    resolved_by: str = Field(..., min_length=1)


"""
Alert API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.database.alerts import AlertConfig, Alert, AlertType, AlertSeverity, AlertStatus
from app.models.schemas.alerts import (
    AlertConfigCreate, AlertConfigResponse, AlertResponse,
    AlertListResponse, AlertAcknowledgeRequest, AlertResolveRequest
)
from app.services.alerting.alert_service import AlertService

router = APIRouter(prefix="/alerts", tags=["Alerts"])


@router.post("/config", response_model=AlertConfigResponse, status_code=status.HTTP_201_CREATED)
async def create_alert_config(
    config: AlertConfigCreate,
    db: AsyncSession = Depends(get_db)
):
    """Create alert configuration."""
    alert_config = await AlertService.create_alert_config(
        db=db,
        name=config.name,
        alert_type=AlertType(config.alert_type),
        conditions=config.conditions,
        notification_channels=config.notification_channels,
        severity=AlertSeverity(config.severity) if config.severity else AlertSeverity.MEDIUM,
        model_id=config.model_id,
        data_source_id=config.data_source_id,
        email_recipients=config.email_recipients,
        webhook_url=config.webhook_url,
        description=config.description
    )
    
    return AlertConfigResponse.model_validate(alert_config)


@router.get("/config", response_model=List[AlertConfigResponse])
async def list_alert_configs(
    enabled: Optional[bool] = None,
    alert_type: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """List alert configurations."""
    from sqlalchemy import select, and_
    
    query = select(AlertConfig)
    conditions = []
    
    if enabled is not None:
        conditions.append(AlertConfig.enabled == enabled)
    if alert_type:
        conditions.append(AlertConfig.alert_type == AlertType(alert_type))
    
    if conditions:
        query = query.where(and_(*conditions))
    
    result = await db.execute(query)
    configs = result.scalars().all()
    
    return [AlertConfigResponse.model_validate(c) for c in configs]


@router.get("/", response_model=AlertListResponse)
async def list_alerts(
    alert_type: Optional[str] = None,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    model_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """List alerts."""
    alerts = await AlertService.get_alerts(
        db=db,
        alert_type=AlertType(alert_type) if alert_type else None,
        status=AlertStatus(status) if status else None,
        severity=AlertSeverity(severity) if severity else None,
        model_id=model_id,
        limit=limit,
        offset=offset
    )
    
    return AlertListResponse(
        alerts=[AlertResponse.model_validate(a) for a in alerts],
        total=len(alerts)
    )


@router.post("/{alert_id}/acknowledge", response_model=AlertResponse)
async def acknowledge_alert(
    alert_id: int,
    request: AlertAcknowledgeRequest,
    db: AsyncSession = Depends(get_db)
):
    """Acknowledge an alert."""
    alert = await AlertService.acknowledge_alert(
        db=db,
        alert_id=alert_id,
        acknowledged_by=request.acknowledged_by
    )
    
    return AlertResponse.model_validate(alert)


@router.post("/{alert_id}/resolve", response_model=AlertResponse)
async def resolve_alert(
    alert_id: int,
    request: AlertResolveRequest,
    db: AsyncSession = Depends(get_db)
):
    """Resolve an alert."""
    alert = await AlertService.resolve_alert(
        db=db,
        alert_id=alert_id,
        resolved_by=request.resolved_by
    )
    
    return AlertResponse.model_validate(alert)


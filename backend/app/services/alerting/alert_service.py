"""
Alert service for managing alerts.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.core.logging import get_logger
from app.models.database.alerts import (
    AlertConfig, Alert, AlertType, AlertSeverity, AlertStatus
)
from app.services.alerting.notifiers import email_notifier, webhook_notifier, in_app_notifier
from app.pipelines.monitoring import ModelPerformanceMonitor

logger = get_logger(__name__)


class AlertService:
    """Service for managing alerts."""
    
    @staticmethod
    async def create_alert_config(
        db: AsyncSession,
        name: str,
        alert_type: AlertType,
        conditions: Dict[str, Any],
        notification_channels: List[str],
        severity: AlertSeverity = AlertSeverity.MEDIUM,
        model_id: Optional[int] = None,
        data_source_id: Optional[int] = None,
        email_recipients: Optional[List[str]] = None,
        webhook_url: Optional[str] = None,
        description: Optional[str] = None
    ) -> AlertConfig:
        """
        Create alert configuration.
        
        Args:
            db: Database session
            name: Alert config name
            alert_type: Type of alert
            conditions: Alert conditions (e.g., {"threshold": 0.05, "metric": "accuracy"})
            notification_channels: List of channels (email, webhook, in_app)
            severity: Alert severity
            model_id: Associated model ID
            data_source_id: Associated data source ID
            email_recipients: Email recipients
            webhook_url: Webhook URL
            description: Description
            
        Returns:
            Created alert config
        """
        config = AlertConfig(
            name=name,
            alert_type=alert_type,
            severity=severity,
            conditions=conditions,
            notification_channels=notification_channels,
            email_recipients=email_recipients or [],
            webhook_url=webhook_url,
            model_id=model_id,
            data_source_id=data_source_id,
            description=description
        )
        
        db.add(config)
        await db.commit()
        await db.refresh(config)
        
        logger.info(f"Created alert config: {name}")
        return config
    
    @staticmethod
    async def check_and_trigger_alerts(
        db: AsyncSession,
        alert_type: AlertType,
        context: Dict[str, Any]
    ) -> List[Alert]:
        """
        Check alert conditions and trigger alerts if needed.
        
        Args:
            db: Database session
            alert_type: Type of alert to check
            context: Context data (model_id, metric_value, etc.)
            
        Returns:
            List of triggered alerts
        """
        # Get enabled alert configs for this type
        query = select(AlertConfig).where(
            and_(
                AlertConfig.alert_type == alert_type,
                AlertConfig.enabled == True
            )
        )
        
        # Filter by model_id or data_source_id if provided
        if context.get("model_id"):
            query = query.where(
                or_(
                    AlertConfig.model_id == context["model_id"],
                    AlertConfig.model_id.is_(None)
                )
            )
        
        if context.get("data_source_id"):
            query = query.where(
                or_(
                    AlertConfig.data_source_id == context["data_source_id"],
                    AlertConfig.data_source_id.is_(None)
                )
            )
        
        result = await db.execute(query)
        configs = result.scalars().all()
        
        triggered_alerts = []
        
        for config in configs:
            should_trigger = await AlertService._evaluate_conditions(
                config, context
            )
            
            if should_trigger:
                alert = await AlertService._create_alert(db, config, context)
                triggered_alerts.append(alert)
                
                # Send notifications
                await AlertService._send_notifications(db, alert, config)
        
        return triggered_alerts
    
    @staticmethod
    async def _evaluate_conditions(
        config: AlertConfig,
        context: Dict[str, Any]
    ) -> bool:
        """
        Evaluate alert conditions.
        
        Args:
            config: Alert config
            context: Context data
            
        Returns:
            True if conditions are met
        """
        conditions = config.conditions
        
        if config.alert_type == AlertType.MODEL_PERFORMANCE:
            # Check performance degradation
            metric = conditions.get("metric", "accuracy")
            threshold = conditions.get("threshold", 0.05)
            current_value = context.get("metric_value")
            baseline_value = context.get("baseline_value")
            
            if current_value is None or baseline_value is None:
                return False
            
            if metric in ["rmse", "mae"]:
                degradation = (current_value - baseline_value) / baseline_value
            else:
                degradation = (baseline_value - current_value) / baseline_value
            
            return degradation > threshold
        
        elif config.alert_type == AlertType.ANOMALY_DETECTION:
            # Check anomaly score
            threshold = conditions.get("threshold", 0.7)
            anomaly_score = context.get("anomaly_score")
            
            if anomaly_score is None:
                return False
            
            return anomaly_score > threshold
        
        elif config.alert_type == AlertType.PIPELINE_FAILURE:
            # Check if pipeline failed
            return context.get("failed", False)
        
        elif config.alert_type == AlertType.DATA_QUALITY:
            # Check data quality metrics
            threshold = conditions.get("threshold", 0.1)
            quality_score = context.get("quality_score", 1.0)
            
            return quality_score < (1.0 - threshold)
        
        return False
    
    @staticmethod
    async def _create_alert(
        db: AsyncSession,
        config: AlertConfig,
        context: Dict[str, Any]
    ) -> Alert:
        """
        Create alert instance.
        
        Args:
            db: Database session
            config: Alert config
            context: Context data
            
        Returns:
            Created alert
        """
        # Generate title and message
        title = f"{config.alert_type.value.replace('_', ' ').title()} Alert: {config.name}"
        message = context.get("message", f"Alert condition met for {config.name}")
        
        alert = Alert(
            config_id=config.id,
            alert_type=config.alert_type,
            severity=config.severity,
            title=title,
            message=message,
            details=context,
            model_id=context.get("model_id"),
            data_source_id=context.get("data_source_id"),
            etl_job_id=context.get("etl_job_id")
        )
        
        db.add(alert)
        await db.commit()
        await db.refresh(alert)
        
        logger.info(f"Created alert: {alert.id} for config: {config.name}")
        return alert
    
    @staticmethod
    async def _send_notifications(
        db: AsyncSession,
        alert: Alert,
        config: AlertConfig
    ):
        """
        Send notifications for alert.
        
        Args:
            db: Database session
            alert: Alert instance
            config: Alert config
        """
        notifications_sent = []
        
        # Email notifications
        if "email" in config.notification_channels and config.email_recipients:
            subject = f"[{alert.severity.value.upper()}] {alert.title}"
            body = f"{alert.message}\n\nDetails: {alert.details}"
            
            success = await email_notifier.send(
                recipients=config.email_recipients,
                subject=subject,
                body=body
            )
            
            if success:
                notifications_sent.append("email")
        
        # Webhook notifications
        if "webhook" in config.notification_channels and config.webhook_url:
            payload = {
                "alert_id": alert.id,
                "type": alert.alert_type.value,
                "severity": alert.severity.value,
                "title": alert.title,
                "message": alert.message,
                "details": alert.details,
                "created_at": alert.created_at.isoformat() if alert.created_at else None
            }
            
            success = await webhook_notifier.send(
                url=config.webhook_url,
                payload=payload
            )
            
            if success:
                notifications_sent.append("webhook")
        
        # In-app notifications
        if "in_app" in config.notification_channels:
            await in_app_notifier.send(db, alert.id)
            notifications_sent.append("in_app")
        
        # Update alert with notification status
        alert.notifications_sent = notifications_sent
        alert.status = AlertStatus.SENT
        await db.commit()
    
    @staticmethod
    async def get_alerts(
        db: AsyncSession,
        alert_type: Optional[AlertType] = None,
        status: Optional[AlertStatus] = None,
        severity: Optional[AlertSeverity] = None,
        model_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Alert]:
        """
        Get alerts with filters.
        
        Args:
            db: Database session
            alert_type: Filter by alert type
            status: Filter by status
            severity: Filter by severity
            model_id: Filter by model ID
            limit: Limit results
            offset: Offset results
            
        Returns:
            List of alerts
        """
        query = select(Alert)
        
        conditions = []
        if alert_type:
            conditions.append(Alert.alert_type == alert_type)
        if status:
            conditions.append(Alert.status == status)
        if severity:
            conditions.append(Alert.severity == severity)
        if model_id:
            conditions.append(Alert.model_id == model_id)
        
        if conditions:
            query = query.where(and_(*conditions))
        
        query = query.order_by(Alert.created_at.desc()).limit(limit).offset(offset)
        
        result = await db.execute(query)
        return result.scalars().all()
    
    @staticmethod
    async def acknowledge_alert(
        db: AsyncSession,
        alert_id: int,
        acknowledged_by: str
    ) -> Alert:
        """
        Acknowledge an alert.
        
        Args:
            db: Database session
            alert_id: Alert ID
            acknowledged_by: User who acknowledged
            
        Returns:
            Updated alert
        """
        alert = await db.get(Alert, alert_id)
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now(timezone.utc)
        
        await db.commit()
        await db.refresh(alert)
        
        return alert
    
    @staticmethod
    async def resolve_alert(
        db: AsyncSession,
        alert_id: int,
        resolved_by: str
    ) -> Alert:
        """
        Resolve an alert.
        
        Args:
            db: Database session
            alert_id: Alert ID
            resolved_by: User who resolved
            
        Returns:
            Updated alert
        """
        alert = await db.get(Alert, alert_id)
        if not alert:
            raise ValueError(f"Alert {alert_id} not found")
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_by = resolved_by
        alert.resolved_at = datetime.now(timezone.utc)
        
        await db.commit()
        await db.refresh(alert)
        
        return alert


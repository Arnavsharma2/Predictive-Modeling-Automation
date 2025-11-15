"""
Model performance monitoring service.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelStatus
from app.models.database.model_versions import ModelVersion

logger = get_logger(__name__)


class ModelPerformanceMonitor:
    """Monitor model performance over time."""
    
    @staticmethod
    async def check_model_performance(
        model_id: int,
        baseline_metric: str = "accuracy",
        degradation_threshold: float = 0.05  # 5% degradation
    ) -> Dict[str, Any]:
        """
        Check if model performance has degraded.
        
        Args:
            model_id: Model ID to check
            baseline_metric: Metric to compare (accuracy, r2_score, etc.)
            degradation_threshold: Threshold for performance degradation
            
        Returns:
            Dictionary with performance check results
        """
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            model = await session.get(MLModel, model_id)
            if not model:
                return {"error": "Model not found"}
            
            # Get active version
            active_version = await VersionManager.get_active_version(session, model_id)
            if not active_version:
                return {"error": "No active version found"}
            
            # Get baseline (from model or first version)
            baseline_value = None
            if baseline_metric == "accuracy" and model.accuracy:
                baseline_value = model.accuracy
            elif baseline_metric == "r2_score" and model.r2_score:
                baseline_value = model.r2_score
            elif baseline_metric == "rmse" and model.rmse:
                baseline_value = model.rmse
            elif active_version.performance_metrics:
                baseline_value = active_version.performance_metrics.get(baseline_metric)
            
            if baseline_value is None:
                return {"error": f"Baseline {baseline_metric} not found"}
            
            # Get current performance (would come from prediction tracking)
            # For now, we'll use the active version's metrics
            current_value = None
            if active_version.performance_metrics:
                current_value = active_version.performance_metrics.get(baseline_metric)
            
            if current_value is None:
                return {"error": f"Current {baseline_metric} not found"}
            
            # Calculate degradation
            if baseline_metric in ["rmse", "mae"]:
                # For these metrics, higher is worse
                degradation = (current_value - baseline_value) / baseline_value
                is_degraded = degradation > degradation_threshold
            else:
                # For accuracy, r2_score, etc., lower is worse
                degradation = (baseline_value - current_value) / baseline_value
                is_degraded = degradation > degradation_threshold
            
            return {
                "model_id": model_id,
                "baseline_metric": baseline_metric,
                "baseline_value": baseline_value,
                "current_value": current_value,
                "degradation": degradation,
                "is_degraded": is_degraded,
                "threshold": degradation_threshold
            }
    
    @staticmethod
    async def get_performance_history(
        model_id: int,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Get model performance history.
        
        Args:
            model_id: Model ID
            days: Number of days to look back
            
        Returns:
            List of performance snapshots
        """
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            # Get all versions created in the last N days
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = select(ModelVersion).where(
                and_(
                    ModelVersion.model_id == model_id,
                    ModelVersion.created_at >= cutoff_date
                )
            ).order_by(ModelVersion.created_at.desc())
            
            result = await session.execute(query)
            versions = result.scalars().all()
            
            history = []
            for version in versions:
                history.append({
                    "version": version.version,
                    "date": version.created_at.isoformat() if version.created_at else None,
                    "metrics": version.performance_metrics or {},
                    "is_active": version.is_active
                })
            
            return history
    
    @staticmethod
    async def compare_with_baseline(
        model_id: int,
        baseline_version_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compare current model performance with baseline.
        
        Args:
            model_id: Model ID
            baseline_version_id: Baseline version ID (if None, uses first version)
            
        Returns:
            Comparison results
        """
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            model = await session.get(MLModel, model_id)
            if not model:
                return {"error": "Model not found"}
            
            # Get baseline version
            if baseline_version_id:
                baseline_version = await VersionManager.get_version(session, baseline_version_id)
            else:
                # Get first version
                versions = await VersionManager.get_versions_by_model(session, model_id)
                baseline_version = versions[-1] if versions else None
            
            # Get active version
            active_version = await VersionManager.get_active_version(session, model_id)
            
            if not baseline_version or not active_version:
                return {"error": "Baseline or active version not found"}
            
            comparison = {}
            baseline_metrics = baseline_version.performance_metrics or {}
            active_metrics = active_version.performance_metrics or {}
            
            # Compare all metrics
            all_metrics = set(list(baseline_metrics.keys()) + list(active_metrics.keys()))
            for metric in all_metrics:
                baseline_val = baseline_metrics.get(metric)
                active_val = active_metrics.get(metric)
                
                if baseline_val is not None and active_val is not None:
                    if metric in ["rmse", "mae"]:
                        change = active_val - baseline_val
                        change_pct = (change / baseline_val) * 100
                    else:
                        change = active_val - baseline_val
                        change_pct = (change / baseline_val) * 100 if baseline_val != 0 else 0
                    
                    comparison[metric] = {
                        "baseline": baseline_val,
                        "current": active_val,
                        "change": change,
                        "change_percent": change_pct
                    }
            
            return {
                "model_id": model_id,
                "baseline_version": baseline_version.version,
                "active_version": active_version.version,
                "comparison": comparison
            }


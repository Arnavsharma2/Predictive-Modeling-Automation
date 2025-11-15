"""
Automated model retraining pipeline using Prefect.
"""
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from prefect import flow, task, get_run_logger
from croniter import croniter
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db_session
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel, ModelStatus, ModelType
from app.models.database.data_sources import DataSource
from app.services.training_job import TrainingJobService
from app.ml.versioning.version_manager import VersionManager

logger = get_logger(__name__)


@task(name="check_retraining_conditions")
async def check_retraining_conditions(
    model_id: int,
    new_data_threshold: Optional[int] = None,
    performance_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Check if model should be retrained based on conditions.
    
    Args:
        model_id: Model ID to check
        new_data_threshold: Minimum new data points to trigger retraining
        performance_threshold: Minimum performance drop to trigger retraining
        
    Returns:
        Dictionary with retraining decision and reasons
    """
    # Use get_db_session to ensure proper event loop handling in Prefect tasks
    async with get_db_session() as session:
        model = await session.get(MLModel, model_id)
        if not model:
            return {"should_retrain": False, "reason": "Model not found"}
        
        reasons = []
        should_retrain = False
        
        # Check new data volume
        if new_data_threshold and model.data_source_id:
            from sqlalchemy import select, func
            from app.models.database.data_points import DataPoint
            
            # Count new data points since last training
            last_training = model.updated_at or model.created_at
            query = select(func.count(DataPoint.id)).where(
                DataPoint.source_id == model.data_source_id,
                DataPoint.created_at > last_training
            )
            result = await session.execute(query)
            new_data_count = result.scalar() or 0
            
            if new_data_count >= new_data_threshold:
                should_retrain = True
                reasons.append(f"New data threshold met: {new_data_count} >= {new_data_threshold}")
        
        # Check performance degradation (would need prediction history)
        # For now, we'll skip this check
        
        return {
            "should_retrain": should_retrain,
            "reasons": reasons,
            "model_id": model_id
        }


@task(name="retrain_model")
async def retrain_model_task(
    model_id: int,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Retrain a model.
    
    Args:
        model_id: Model ID to retrain
        training_config: Training configuration
        
    Returns:
        Training results
    """
    # Use get_db_session to ensure proper event loop handling in Prefect tasks
    async with get_db_session() as session:
        model = await session.get(MLModel, model_id)
        if not model:
            raise ValueError(f"Model {model_id} not found")
        
        if not model.data_source_id:
            raise ValueError(f"Model {model_id} has no data source")
        
        # Get target column from model metadata or config
        target_column = training_config.get("target_column") if training_config else None
        if not target_column:
            # Try to infer from model features or use default
            target_column = "target"  # Default, should be configured
        
        # Create training job
        job = await TrainingJobService.create_training_job(
            db=session,
            model_type=model.type.value,
            data_source_id=model.data_source_id,
            model_id=model_id,
            hyperparameters=model.hyperparameters,
            training_config=training_config or {}
        )
        
        # Execute training
        new_model = await TrainingJobService.execute_training_job(
            db=session,
            job_id=job.id,
            data_source_id=model.data_source_id,
            target_column=target_column,
            model_name=f"{model.name}_retrained",
            algorithm=training_config.get("algorithm", "random_forest") if training_config else "random_forest",
            hyperparameters=model.hyperparameters,
            training_config=training_config or {}
        )
        
        return {
            "job_id": job.id,
            "new_model_id": new_model.id,
            "status": "completed"
        }


@flow(name="automated_retraining", retries=1, retry_delay_seconds=60)
async def automated_retraining_flow(
    model_id: int,
    retraining_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Automated retraining flow.
    
    Args:
        model_id: Model ID to retrain
        retraining_config: Retraining configuration
        
    Returns:
        Flow execution results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting automated retraining flow for model {model_id}")
    
    retraining_config = retraining_config or {}
    
    # Check conditions
    conditions = await check_retraining_conditions(
        model_id=model_id,
        new_data_threshold=retraining_config.get("new_data_threshold"),
        performance_threshold=retraining_config.get("performance_threshold")
    )
    
    if not conditions["should_retrain"]:
        flow_logger.info(f"Retraining conditions not met for model {model_id}: {conditions.get('reason')}")
        return {
            "retrained": False,
            "reason": conditions.get("reason", "Conditions not met")
        }
    
    flow_logger.info(f"Retraining conditions met: {conditions['reasons']}")
    
    # Retrain model
    result = await retrain_model_task(
        model_id=model_id,
        training_config=retraining_config.get("training_config")
    )
    
    flow_logger.info(f"Model {model_id} retraining completed. New model ID: {result['new_model_id']}")
    
    return {
        "retrained": True,
        "job_id": result["job_id"],
        "new_model_id": result["new_model_id"]
    }


# Scheduled retraining flows
@flow(name="scheduled_daily_retraining")
async def scheduled_daily_retraining():
    """Scheduled daily retraining for all active models."""
    flow_logger = get_run_logger()
    flow_logger.info("Starting scheduled daily retraining")
    
    # Use get_db_session to ensure proper event loop handling in Prefect tasks
    async with get_db_session() as session:
        from sqlalchemy import select
        
        # Get all trained/deployed models
        query = select(MLModel).where(
            MLModel.status.in_([ModelStatus.TRAINED, ModelStatus.DEPLOYED]),
            MLModel.is_active == True
        )
        result = await session.execute(query)
        models = result.scalars().all()
        
        results = []
        for model in models:
            try:
                result = await automated_retraining_flow(
                    model_id=model.id,
                    retraining_config={"new_data_threshold": 1000}
                )
                results.append({"model_id": model.id, **result})
            except Exception as e:
                flow_logger.error(f"Error retraining model {model.id}: {e}")
                results.append({"model_id": model.id, "retrained": False, "error": str(e)})
        
        return {"results": results}


# Schedule configuration (for Prefect 2.x, use deployments instead)
# To schedule: prefect deployment build app/pipelines/retraining_pipeline.py:scheduled_daily_retraining --cron "0 2 * * *"


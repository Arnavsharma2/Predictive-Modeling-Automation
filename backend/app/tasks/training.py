"""
Celery tasks for model training.
"""
import asyncio
from typing import Dict, Any, Optional
from celery.utils.log import get_task_logger

from app.core.celery_app import celery_app
from app.core.database import get_async_session_context
from app.services.training_job import TrainingJobService
from app.models.database.training_jobs import TrainingJobStatus

logger = get_task_logger(__name__)


async def _execute_training_async(
    job_id: int,
    data_source_id: int,
    target_column: str,
    model_name: str,
    algorithm: str,
    hyperparameters: Optional[Dict[str, Any]],
    training_config: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Async function to execute training job.

    This is defined at module level to avoid AST parsing issues.
    """
    try:
        # Create new database session for this worker
        async with get_async_session_context() as db:
            # Execute training job
            model = await TrainingJobService.execute_training_job(
                db=db,
                job_id=job_id,
                data_source_id=data_source_id,
                target_column=target_column,
                model_name=model_name,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                training_config=training_config
            )

            logger.info(f"Training completed successfully for job {job_id}. Model ID: {model.id}")

            return {
                "model_id": model.id,
                "model_name": model.name,
                "status": "completed",
                "job_id": job_id
            }

    except Exception as e:
        logger.error(f"Error in training task for job {job_id}: {e}", exc_info=True)

        # Try to update job status to failed
        try:
            async with get_async_session_context() as db:
                await TrainingJobService.update_job_status(
                    db, job_id, TrainingJobStatus.FAILED, error_message=str(e)
                )
        except Exception as update_error:
            logger.error(f"Failed to update job status: {update_error}")

        # Re-raise to mark Celery task as failed
        raise


@celery_app.task(
    bind=True,
    name="app.tasks.training.train_model",
    track_started=True,
    acks_late=True
)
def train_model_task(
    self,
    job_id: int,
    data_source_id: int,
    target_column: str,
    model_name: str,
    algorithm: str = "random_forest",
    hyperparameters: Optional[Dict[str, Any]] = None,
    training_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Celery task to train a model in the background.

    This runs in a separate worker process, keeping the main FastAPI
    backend responsive during training.

    Args:
        job_id: Training job ID
        data_source_id: Data source ID
        target_column: Target column name
        model_name: Name for the model
        algorithm: ML algorithm to use
        hyperparameters: Model hyperparameters
        training_config: Training configuration

    Returns:
        Dictionary with model_id and status
    """
    logger.info(f"Starting Celery training task for job {job_id}")
    logger.info(f"Worker: {self.request.hostname}, Task ID: {self.request.id}")

    # Get or create event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the async function
    return loop.run_until_complete(
        _execute_training_async(
            job_id=job_id,
            data_source_id=data_source_id,
            target_column=target_column,
            model_name=model_name,
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            training_config=training_config
        )
    )

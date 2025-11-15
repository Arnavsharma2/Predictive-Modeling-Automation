"""
Batch prediction pipeline using Prefect.
"""
from typing import Dict, Any
from datetime import datetime

from prefect import flow, get_run_logger, task
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.database import get_db_session
from app.models.database.batch_jobs import BatchPredictionJob, BatchJobStatus
from app.services.batch_prediction_job import BatchPredictionJobService

logger = get_logger(__name__)


@task(name="execute_batch_prediction_task", retries=1, retry_delay_seconds=30)
async def execute_batch_prediction_task(
    job_id: int,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Task to execute batch prediction.
    
    Args:
        job_id: Batch prediction job ID
        batch_size: Number of records to process per batch
        
    Returns:
        Dictionary with execution results
    """
    task_logger = get_run_logger()
    task_logger.info(f"Executing batch prediction job {job_id}")
    
    # Use get_db_session to ensure proper event loop handling in Prefect tasks
    async with get_db_session() as session:
        try:
            job = await BatchPredictionJobService.execute_batch_prediction(
                session, job_id, batch_size=batch_size
            )
            
            return {
                "success": True,
                "job_id": job_id,
                "status": job.status.value,
                "processed_records": job.processed_records,
                "failed_records": job.failed_records,
                "result_path": job.result_path
            }
        except Exception as e:
            task_logger.error(f"Batch prediction task failed: {e}")
            raise


@flow(name="batch_prediction_pipeline", retries=1, retry_delay_seconds=30)
async def batch_prediction_pipeline(
    job_id: int,
    batch_size: int = 1000
) -> Dict[str, Any]:
    """
    Main batch prediction pipeline flow.
    
    Args:
        job_id: Batch prediction job ID
        batch_size: Number of records to process per batch
        
    Returns:
        Dictionary with pipeline execution results
    """
    flow_logger = get_run_logger()
    flow_logger.info(f"Starting batch prediction pipeline for job {job_id}")
    
    prefect_flow_run_id = None
    
    try:
        # Get flow run ID (Prefect 2.x compatible)
        try:
            from prefect import runtime
            if hasattr(runtime, 'flow_run') and runtime.flow_run:
                prefect_flow_run_id = str(runtime.flow_run.id)
        except (ImportError, AttributeError):
            prefect_flow_run_id = None
        
        # Update job with Prefect flow run ID
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        async with get_db_session() as session:
            job = await BatchPredictionJobService.get_batch_job(session, job_id)
            if job:
                job.prefect_flow_run_id = prefect_flow_run_id
                await session.commit()
        
        # Execute batch prediction
        result = await execute_batch_prediction_task(job_id, batch_size)
        
        flow_logger.info(f"Batch prediction pipeline completed successfully for job {job_id}")
        return result
        
    except Exception as e:
        flow_logger.error(f"Batch prediction pipeline failed: {e}")
        
        # Update job status to failed
        # Use get_db_session to ensure proper event loop handling in Prefect tasks
        try:
            async with get_db_session() as session:
                await BatchPredictionJobService.update_job_status(
                    session, job_id, BatchJobStatus.FAILED,
                    error_message=str(e)
                )
        except Exception:
            pass
        
        raise


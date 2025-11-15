"""
ML model training API endpoints.
"""
import asyncio
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db, get_db_session
from app.core.logging import get_logger
from app.tasks.training import train_model_task
from app.models.database.ml_models import MLModel, ModelType, ModelStatus
from app.models.database.training_jobs import TrainingJob, TrainingJobStatus
from app.models.database.data_sources import DataSource
from app.models.database.users import User
from app.middleware.auth import get_current_user
from app.core.permissions import has_resource_access, filter_by_user_access
from typing import Annotated
from app.models.schemas.ml import (
    TrainingRequest,
    TrainingResponse,
    TrainingJobResponse,
    TrainingJobListResponse,
    JobCancellationResponse,
    MLModelResponse,
    MLModelListResponse,
    ModelDetailResponse,
    ModelVersionResponse,
    ModelVersionListResponse,
    ModelComparisonResponse,
    ModelRollbackRequest,
    ModelRollbackResponse
)
from app.models.database.model_versions import ModelVersion
from app.services.training_job import TrainingJobService
from app.ml.versioning.version_manager import VersionManager
from app.ml.versioning.comparison import ModelComparison
from app.ml.storage.model_storage import model_storage
from app.storage.cloud_storage import cloud_storage

logger = get_logger(__name__)

router = APIRouter(prefix="/train", tags=["Training"])


async def _run_training_job_in_background(
    job_id: int,
    data_source_id: int,
    target_column: str,
    model_name: str,
    algorithm: str,
    hyperparameters: Optional[dict],
    training_config: Optional[dict]
):
    """
    Wrapper function to run training job in background with its own database session.
    
    This is necessary because the original database session will be closed when the request completes.
    """
    import sys
    print("="*100, file=sys.stderr, flush=True)
    print(f"BACKGROUND TRAINING TASK STARTED - JOB_ID: {job_id}", file=sys.stderr, flush=True)
    print("="*100, file=sys.stderr, flush=True)
    logger.info("="*100)
    logger.info(f"BACKGROUND TRAINING TASK STARTED - JOB_ID: {job_id}")
    logger.info("="*100)
    
    try:
        logger.info(f"Creating database session for training job {job_id}...")
        async with get_db_session() as db:
            logger.info(f"Database session created, calling execute_training_job for job {job_id}...")
            await TrainingJobService.execute_training_job(
                db=db,
                job_id=job_id,
                data_source_id=data_source_id,
                target_column=target_column,
                model_name=model_name,
                algorithm=algorithm,
                hyperparameters=hyperparameters,
                training_config=training_config
            )
            logger.info(f"Training job {job_id} completed successfully")
    except Exception as e:
        logger.error(f"Background training job {job_id} failed: {e}", exc_info=True)
        import sys
        print(f"Background training job {job_id} failed: {e}", file=sys.stderr, flush=True)
        raise


@router.post("", response_model=TrainingResponse, status_code=status.HTTP_201_CREATED)
async def train_model(
    request: TrainingRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Trigger a model training job.
    """
    try:
        # Validate data source exists and user has access
        data_source = await db.get(DataSource, request.data_source_id)
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data source {request.data_source_id} not found"
            )
        
        # Check user has access to data source
        if not has_resource_access(current_user, data_source.created_by, data_source.shared_with):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this data source"
            )
        
        # Validate model type
        if request.model_type not in ["regression", "classification", "anomaly_detection"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model type: {request.model_type}"
            )
        
        # Ensure target_column is stored in training_config for later retrieval
        training_config = request.training_config or {}
        if "target_column" not in training_config:
            training_config = {**training_config, "target_column": request.target_column}
        
        # Create training job with user ownership
        job = await TrainingJobService.create_training_job(
            db=db,
            model_type=request.model_type,
            data_source_id=request.data_source_id,
            hyperparameters=request.hyperparameters,
            training_config=training_config,
            created_by=current_user.id
        )
        
        # Start training in background using Celery worker
        # This runs in a separate process, keeping the FastAPI backend responsive
        # even during CPU-intensive training operations
        logger.info(f"Dispatching training job {job.id} to Celery worker...")

        # Dispatch task to Celery
        celery_task = train_model_task.delay(
            job_id=job.id,
            data_source_id=request.data_source_id,
            target_column=request.target_column,
            model_name=request.model_name,
            algorithm=request.algorithm,
            hyperparameters=request.hyperparameters,
            training_config=request.training_config
        )

        logger.info(f"Celery task dispatched: {celery_task.id}, training job {job.id} for model '{request.model_name}'")
        
        return TrainingResponse(
            job_id=job.id,
            status=job.status,
            message=f"Training job {job.id} started successfully"
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting training job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error starting training job: {str(e)}"
        )


@router.get("/status/{job_id}", response_model=TrainingJobResponse)
async def get_training_status(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Get training job status.
    """
    # Expire any cached objects to ensure we get fresh data
    # This is important because the training job is updated in a background task
    # with a different database session
    db.expire_all()
    
    job = await TrainingJobService.get_training_job(db, job_id)
    
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    # Check user has access to training job
    if not has_resource_access(current_user, job.created_by, None):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this training job"
        )
    
    # Ensure all attributes are loaded while session is open
    # Access all attributes to ensure they're loaded before session closes
    _ = job.id, job.model_id, job.data_source_id, job.status, job.model_type
    _ = job.progress, job.current_epoch, job.total_epochs
    _ = job.started_at, job.completed_at, job.error_message, job.metrics
    _ = job.created_at, job.created_by
    
    # Convert to dict first to ensure all data is in Python primitives
    # This prevents MissingGreenlet errors when session closes
    job_dict = {
        "id": job.id,
        "model_id": job.model_id,
        "data_source_id": job.data_source_id,
        "status": job.status,
        "model_type": job.model_type,
        "progress": job.progress,
        "current_epoch": job.current_epoch,
        "total_epochs": job.total_epochs,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
        "error_message": job.error_message,
        "metrics": job.metrics,
        "created_at": job.created_at,
    }
    
    return TrainingJobResponse.model_validate(job_dict)


@router.get("/history", response_model=TrainingJobListResponse)
async def get_training_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_id: Optional[int] = None,
    status: Optional[TrainingJobStatus] = None,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get training job history.
    """
    query = select(TrainingJob)
    
    # Filter by user access
    query = filter_by_user_access(query, TrainingJob, current_user)
    
    if model_id:
        query = query.where(TrainingJob.model_id == model_id)
    
    if status:
        query = query.where(TrainingJob.status == status)
    
    # Get total count (with same filters)
    count_query = select(func.count()).select_from(TrainingJob)
    count_query = filter_by_user_access(count_query, TrainingJob, current_user)
    if model_id:
        count_query = count_query.where(TrainingJob.model_id == model_id)
    if status:
        count_query = count_query.where(TrainingJob.status == status)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get jobs
    query = query.order_by(TrainingJob.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    # Convert jobs to dicts while session is still open to prevent MissingGreenlet errors
    job_dicts = []
    for job in jobs:
        job_dict = {
            "id": job.id,
            "model_id": job.model_id,
            "data_source_id": job.data_source_id,
            "status": job.status,
            "model_type": job.model_type,
            "progress": job.progress,
            "current_epoch": job.current_epoch,
            "total_epochs": job.total_epochs,
            "started_at": job.started_at,
            "completed_at": job.completed_at,
            "error_message": job.error_message,
            "metrics": job.metrics,
            "created_at": job.created_at,
        }
        job_dicts.append(job_dict)
    
    return TrainingJobListResponse(
        jobs=[TrainingJobResponse.model_validate(job_dict) for job_dict in job_dicts],
        total=total
    )




@router.post("/train/{job_id}/cancel", response_model=JobCancellationResponse)
async def cancel_training_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
) -> JobCancellationResponse:
    """
    Cancel a running or pending training job.

    Args:
        job_id: Training job ID to cancel
        current_user: Current authenticated user
        db: Database session

    Returns:
        Success message
    """
    from app.services.job_cleanup import JobCleanupService

    # Get the job
    job = await TrainingJobService.get_training_job(db, job_id)

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )

    # Check user has access to this job
    if not has_resource_access(current_user, job.created_by, None):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to cancel this training job"
        )

    # Cancel the job
    success = await JobCleanupService.cancel_job(
        db,
        job_id,
        f"Cancelled by user {current_user.username}"
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job {job_id}. Job may not be running or pending."
        )

    return JobCancellationResponse(
        message=f"Training job {job_id} has been cancelled",
        job_id=job_id,
        status="CANCELLED"
    )

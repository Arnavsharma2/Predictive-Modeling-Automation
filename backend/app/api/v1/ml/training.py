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
    skip: int = 0,
    limit: int = 100,
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


@router.get("/models", response_model=MLModelListResponse)
async def list_models(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
    model_type: Optional[ModelType] = None,
    status: Optional[ModelStatus] = None
):
    """
    List all ML models (filtered by user access).
    """
    query = select(MLModel)
    
    # Filter by user access
    query = filter_by_user_access(query, MLModel, current_user)
    
    if model_type:
        query = query.where(MLModel.type == model_type)
    
    if status:
        query = query.where(MLModel.status == status)
    
    # Get total count (with same filters)
    count_query = select(func.count()).select_from(MLModel)
    count_query = filter_by_user_access(count_query, MLModel, current_user)
    if model_type:
        count_query = count_query.where(MLModel.type == model_type)
    if status:
        count_query = count_query.where(MLModel.status == status)
    
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Get models
    query = query.order_by(MLModel.created_at.desc()).offset(skip).limit(limit)
    result = await db.execute(query)
    models = result.scalars().all()
    
    return MLModelListResponse(
        models=[MLModelResponse.model_validate(model) for model in models],
        total=total
    )


@router.get("/models/{model_id}", response_model=ModelDetailResponse)
async def get_model_details(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed model information.
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Check user has access to model
    if not has_resource_access(current_user, model.created_by, model.shared_with):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this model"
        )
    
    # Get training history
    training_jobs_query = select(TrainingJob).where(TrainingJob.model_id == model_id)
    training_jobs_result = await db.execute(training_jobs_query)
    training_jobs = training_jobs_result.scalars().all()
    
    training_history = [
        {
            "job_id": job.id,
            "status": job.status.value,
            "metrics": job.metrics,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None
        }
        for job in training_jobs
    ]
    
    response = ModelDetailResponse.model_validate(model)
    response.training_history = training_history
    
    return response


@router.get("/models/{model_id}/versions", response_model=ModelVersionListResponse)
async def list_model_versions(
    model_id: int,
    include_archived: bool = False,
    db: AsyncSession = Depends(get_db)
):
    """
    List all versions for a model.
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    versions = await VersionManager.get_versions_by_model(db, model_id, include_archived=include_archived)
    
    return ModelVersionListResponse(
        versions=[ModelVersionResponse.model_validate(v) for v in versions],
        total=len(versions)
    )


@router.get("/models/{model_id}/compare", response_model=ModelComparisonResponse)
async def compare_model_versions(
    model_id: int,
    version_ids: Optional[str] = None,  # Comma-separated version IDs
    db: AsyncSession = Depends(get_db)
):
    """
    Compare model versions.
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Get versions to compare
    if version_ids:
        version_id_list = [int(vid.strip()) for vid in version_ids.split(",")]
        versions = []
        for vid in version_id_list:
            version = await VersionManager.get_version(db, vid)
            if version and version.model_id == model_id:
                versions.append(version)
    else:
        # Compare all versions
        versions = await VersionManager.get_versions_by_model(db, model_id, include_archived=False)
    
    if len(versions) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Need at least 2 versions to compare"
        )
    
    # Perform comparison
    comparison = ModelComparison.compare_versions(versions)
    
    return ModelComparisonResponse(
        model_id=model_id,
        versions=[ModelVersionResponse.model_validate(v) for v in versions],
        comparison_metrics=comparison
    )


@router.post("/models/{model_id}/rollback", response_model=ModelRollbackResponse)
async def rollback_model(
    model_id: int,
    request: ModelRollbackRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Rollback model to a previous version.
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    # Get current active version
    current_active = await VersionManager.get_active_version(db, model_id)
    previous_version = current_active.version if current_active else None
    
    # Get version to rollback to
    target_version = await VersionManager.get_version(db, request.version_id)
    
    if not target_version:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Version {request.version_id} not found"
        )
    
    if target_version.model_id != model_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Version {request.version_id} does not belong to model {model_id}"
        )
    
    if target_version.is_archived:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot rollback to archived version {target_version.version}"
        )
    
    # Activate the target version
    new_active = await VersionManager.activate_version(db, request.version_id)
    
    # Update model to use the new version's path
    model.model_path = new_active.model_path
    model.version = new_active.version
    
    # Update model metrics from version
    if new_active.performance_metrics:
        if "accuracy" in new_active.performance_metrics:
            model.accuracy = new_active.performance_metrics.get("accuracy")
        if "precision" in new_active.performance_metrics:
            model.precision = new_active.performance_metrics.get("precision")
        if "recall" in new_active.performance_metrics:
            model.recall = new_active.performance_metrics.get("recall")
        if "f1_score" in new_active.performance_metrics:
            model.f1_score = new_active.performance_metrics.get("f1_score")
        if "rmse" in new_active.performance_metrics:
            model.rmse = new_active.performance_metrics.get("rmse")
        if "mae" in new_active.performance_metrics:
            model.mae = new_active.performance_metrics.get("mae")
        if "r2_score" in new_active.performance_metrics:
            model.r2_score = new_active.performance_metrics.get("r2_score")
    
    await db.commit()
    
    logger.info(f"Rolled back model {model_id} from version {previous_version} to {new_active.version}")
    
    return ModelRollbackResponse(
        model_id=model_id,
        previous_version=previous_version or "unknown",
        new_active_version=new_active.version,
        message=f"Successfully rolled back to version {new_active.version}"
    )


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a model and all its associated data.
    This will also delete:
    - All model versions (CASCADE)
    - All training jobs (CASCADE)
    - Model files from storage
    """
    model = await db.get(MLModel, model_id)
    
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found"
        )
    
    try:
        # Get all versions to delete their model files
        versions = await VersionManager.get_versions_by_model(db, model_id, include_archived=True)
        
        # Delete model files from storage
        if model.model_path:
            try:
                await model_storage.delete_model(model.model_path)
                logger.info(f"Deleted model file: {model.model_path}")
            except Exception as e:
                logger.warning(f"Failed to delete model file {model.model_path}: {e}")
        
        # Delete version files from storage
        for version in versions:
            if version.model_path:
                try:
                    await model_storage.delete_model(version.model_path)
                    logger.info(f"Deleted version file: {version.model_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete version file {version.model_path}: {e}")
        
        # Delete the model (CASCADE will handle versions, training jobs, etc.)
        await db.delete(model)
        await db.commit()
        
        logger.info(f"Successfully deleted model {model_id} ({model.name})")
        
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting model: {str(e)}"
        )


@router.delete("/models/bulk", status_code=status.HTTP_204_NO_CONTENT)
async def bulk_delete_models(
    model_ids: str = Query(..., description="Comma-separated list of model IDs to delete"),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete multiple models and all their associated data.
    """
    try:
        if not model_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No model IDs provided"
            )
        
        # Parse comma-separated IDs
        try:
            id_list = [int(id_str.strip()) for id_str in model_ids.split(',') if id_str.strip()]
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid model IDs format. Expected comma-separated integers."
            )
        
        if not id_list:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No valid model IDs provided"
            )
        
        # Get all models
        result = await db.execute(
            select(MLModel)
            .where(MLModel.id.in_(id_list))
        )
        models_to_delete = result.scalars().all()
        
        if len(models_to_delete) != len(id_list):
            found_ids = {m.id for m in models_to_delete}
            missing_ids = set(id_list) - found_ids
            logger.warning(f"Some models not found: {missing_ids}")
        
        # Delete model files and versions for each model
        for model in models_to_delete:
            try:
                # Get all versions to delete their model files
                versions = await VersionManager.get_versions_by_model(db, model.id, include_archived=True)
                
                # Delete model files from storage
                if model.model_path:
                    try:
                        await model_storage.delete_model(model.model_path)
                        logger.info(f"Deleted model file: {model.model_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete model file {model.model_path}: {e}")
                
                # Delete version files from storage
                for version in versions:
                    if version.model_path:
                        try:
                            await model_storage.delete_model(version.model_path)
                            logger.info(f"Deleted version file: {version.model_path}")
                        except Exception as e:
                            logger.warning(f"Failed to delete version file {version.model_path}: {e}")
            except Exception as e:
                logger.warning(f"Error deleting files for model {model.id}: {e}")
        
        # Delete all models (CASCADE will handle versions, training jobs, etc.)
        for model in models_to_delete:
            await db.delete(model)
        
        await db.commit()
        
        logger.info(f"Successfully deleted {len(models_to_delete)} models")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk deleting models: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting models: {str(e)}"
        )


@router.post("/train/{job_id}/cancel")
async def cancel_training_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
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

    return {
        "message": f"Training job {job_id} has been cancelled",
        "job_id": job_id,
        "status": "CANCELLED"
    }

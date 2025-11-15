"""
ML API endpoints.
"""
from typing import Optional
from fastapi import APIRouter, Depends, Query, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.middleware.auth import get_current_user
from app.models.database.ml_models import MLModel, ModelType, ModelStatus
from app.models.database.training_jobs import TrainingJob, TrainingJobStatus
from app.models.database.data_sources import DataSource
from app.models.schemas.ml import (
    MLModelListResponse,
    MLModelResponse,
    TrainingJobListResponse,
    TrainingJobResponse,
    TrainingRequest,
    TrainingResponse,
    ModelDetailResponse,
    ModelVersionListResponse,
    ModelVersionResponse
)
from app.core.permissions import filter_by_user_access, has_resource_access
from typing import Annotated
from app.models.database.users import User
from sqlalchemy import select, func
from app.tasks.training import train_model_task
from app.services.training_job import TrainingJobService
from app.services.drift_monitoring import DriftMonitoringService
from app.models.database.drift_reports import DriftReport
from app.models.schemas.drift import DriftReportResponse, DriftReportListResponse
from app.ml.versioning.version_manager import VersionManager
from app.ml.storage.model_storage import model_storage

logger = get_logger(__name__)

from app.api.v1.ml.training import router as training_router
from app.api.v1.ml.predictions import router as predictions_router
from app.api.v1.ml.anomaly import router as anomaly_router
from app.api.v1.ml.classification import router as classification_router
from app.api.v1.ml.registry import router as registry_router
from app.api.v1.ml.retrain import router as retrain_router
from app.api.v1.ml.explain import router as explain_router
from app.api.v1.ml.experiments import router as experiments_router
from app.api.v1.ml.optimize import router as optimize_router
from app.api.v1.ml.drift import router as drift_router
from app.api.v1.ml.batch_predictions import router as batch_predictions_router

router = APIRouter(prefix="/ml", tags=["ML"])


@router.get("/models", response_model=MLModelListResponse)
async def list_models(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
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


@router.post("/training", response_model=TrainingResponse, status_code=status.HTTP_201_CREATED)
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


@router.get("/training/{job_id}/status", response_model=TrainingJobResponse)
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


@router.get("/training", response_model=TrainingJobListResponse)
async def get_training_history(
    current_user: Annotated[User, Depends(get_current_user)],
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_id: Optional[int] = None,
    status: Optional[TrainingJobStatus] = None,
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


@router.get("/drift/{model_id}/reports", response_model=DriftReportListResponse)
async def get_drift_reports_by_model_id(
    model_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get drift reports for a model (by model ID in path).
    """
    try:
        # Check model access
        model_result = await db.execute(
            select(MLModel).where(MLModel.id == model_id)
        )
        model = model_result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        if not has_resource_access(current_user, model):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this model"
            )
        
        # Get drift reports
        monitoring_service = DriftMonitoringService()
        reports = await monitoring_service.get_drift_history(
            db=db,
            model_id=model_id,
            limit=limit,
            offset=skip
        )
        
        # Get total count
        count_result = await db.execute(
            select(func.count(DriftReport.id)).where(DriftReport.model_id == model_id)
        )
        total = count_result.scalar() or 0
        
        return DriftReportListResponse(
            reports=[DriftReportResponse.from_orm(r) for r in reports],
            total=total,
            limit=limit,
            offset=skip
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting drift reports: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting drift reports: {str(e)}"
        )


@router.get("/drift/{model_id}/latest", response_model=DriftReportResponse)
async def get_latest_drift_report_by_model_id(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get the latest drift report for a model (by model ID in path).
    """
    try:
        # Check model access
        model_result = await db.execute(
            select(MLModel).where(MLModel.id == model_id)
        )
        model = model_result.scalar_one_or_none()
        
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_id} not found"
            )
        
        if not has_resource_access(current_user, model):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this model"
            )
        
        # Get latest drift report
        monitoring_service = DriftMonitoringService()
        drift_report = await monitoring_service.get_latest_drift_report(
            db=db,
            model_id=model_id
        )
        
        if not drift_report:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No drift reports found for this model"
            )
        
        return DriftReportResponse.from_orm(drift_report)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting latest drift report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting latest drift report: {str(e)}"
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
    current_user: Annotated[User, Depends(get_current_user)],
    include_archived: bool = Query(False, description="Include archived versions"),
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
    
    # Check user has access to model
    if not has_resource_access(current_user, model.created_by, model.shared_with):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this model"
        )
    
    versions = await VersionManager.get_versions_by_model(db, model_id, include_archived=include_archived)
    
    return ModelVersionListResponse(
        versions=[ModelVersionResponse.model_validate(v) for v in versions],
        total=len(versions)
    )


@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a model and all its associated data (by model ID in path).
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
    
    # Check user has access to this model
    if not has_resource_access(current_user, model):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You don't have access to this model"
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


router.include_router(training_router)
router.include_router(predictions_router)
router.include_router(anomaly_router)
router.include_router(classification_router)
router.include_router(registry_router)
router.include_router(retrain_router)
router.include_router(explain_router)
router.include_router(experiments_router)
router.include_router(optimize_router)
router.include_router(drift_router)
router.include_router(batch_predictions_router)


"""
Batch prediction job API endpoints.
"""
import asyncio
import uuid
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query, File, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from pydantic import BaseModel

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.batch_jobs import BatchPredictionJob, BatchJobStatus
from app.models.database.ml_models import MLModel
from app.middleware.auth import get_current_user
from app.core.permissions import has_resource_access, filter_by_user_access
from typing import Annotated
from app.models.schemas.ml import (
    BatchJobCreateRequest,
    BatchJobResponse,
    BatchJobListResponse
)
from app.services.batch_prediction_job import BatchPredictionJobService
from app.pipelines.batch_prediction_pipeline import batch_prediction_pipeline
from app.models.database.users import User
from app.storage.cloud_storage import cloud_storage

logger = get_logger(__name__)

router = APIRouter(prefix="/batch-predictions", tags=["Batch Predictions"])


class FileUploadResponse(BaseModel):
    """Response for file upload."""
    file_path: str
    filename: str
    message: str = "File uploaded successfully"


@router.post("/upload-file", response_model=FileUploadResponse, status_code=status.HTTP_200_OK)
async def upload_batch_file(
    file: UploadFile = File(...),
    current_user: Annotated[User, Depends(get_current_user)] = None,
):
    """
    Upload a file for batch prediction input.
    Returns the file path to use in input_config.
    """
    try:
        # Read file content
        file_content = await file.read()
        
        # Validate file is not empty
        if len(file_content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Generate unique file path
        file_id = str(uuid.uuid4())
        file_extension = Path(file.filename).suffix or ".csv"
        file_path = f"batch_inputs/{file_id}{file_extension}"
        
        # Upload to cloud storage or save locally
        try:
            storage_path = await cloud_storage.upload_file(
                file_content=file_content,
                file_path=file_path,
                content_type=file.content_type or "text/csv"
            )
        except Exception as e:
            logger.warning(f"Could not upload to cloud storage, saving locally: {e}")
            # Save file locally
            uploads_dir = Path("uploads/batch_inputs")
            uploads_dir.mkdir(parents=True, exist_ok=True)
            local_file_path = uploads_dir / f"{file_id}{file_extension}"
            with open(local_file_path, "wb") as f:
                f.write(file_content)
            storage_path = str(local_file_path)
            logger.info(f"File saved locally at {storage_path}")
        
        return FileUploadResponse(
            file_path=storage_path,
            filename=file.filename,
            message="File uploaded successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file for batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading file: {str(e)}"
        )


@router.post("", response_model=BatchJobResponse, status_code=status.HTTP_201_CREATED)
async def create_batch_job(
    request: BatchJobCreateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new batch prediction job.
    """
    try:
        # Check if user has access to the model
        model = await db.get(MLModel, request.model_id)
        if not model:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {request.model_id} not found"
            )
        
        if not has_resource_access(current_user, model.created_by, model.shared_with):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this model"
            )
        
        # Create batch job
        job = await BatchPredictionJobService.create_batch_job(
            db=db,
            model_id=request.model_id,
            input_type=request.input_type,
            data_source_id=request.data_source_id,
            input_config=request.input_config,
            job_name=request.job_name,
            result_format=request.result_format,
            scheduled_at=request.scheduled_at,
            created_by=current_user.id
        )
        
        # If not scheduled, start the job immediately
        if not request.scheduled_at:
            # Run in background
            asyncio.create_task(
                batch_prediction_pipeline(job.id, batch_size=1000)
            )
            logger.info(f"Started batch prediction job {job.id} in background")
        
        return BatchJobResponse.model_validate(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating batch prediction job: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating batch prediction job: {str(e)}"
        )


@router.get("", response_model=BatchJobListResponse)
async def list_batch_jobs(
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    model_id: Optional[int] = Query(None),
    status_filter: Optional[str] = Query(None)
):
    """
    List batch prediction jobs.
    """
    try:
        # Build query
        query = select(BatchPredictionJob)
        
        # Filter by user access
        query = filter_by_user_access(query, BatchPredictionJob, current_user)
        
        # Filter by model_id if provided
        if model_id:
            query = query.where(BatchPredictionJob.model_id == model_id)
        
        # Filter by status if provided
        if status_filter:
            try:
                status_enum = BatchJobStatus(status_filter)
                query = query.where(BatchPredictionJob.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status_filter}"
                )
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total_result = await db.execute(count_query)
        total = total_result.scalar()
        
        # Apply pagination
        query = query.order_by(BatchPredictionJob.created_at.desc())
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await db.execute(query)
        jobs = result.scalars().all()
        
        return BatchJobListResponse(
            jobs=[BatchJobResponse.model_validate(job) for job in jobs],
            total=total
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing batch jobs: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing batch jobs: {str(e)}"
        )


@router.get("/{job_id}", response_model=BatchJobResponse)
async def get_batch_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Get batch prediction job by ID.
    """
    try:
        job = await BatchPredictionJobService.get_batch_job(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch prediction job {job_id} not found"
            )
        
        # Check access
        model = await db.get(MLModel, job.model_id)
        if model and not has_resource_access(current_user, model.created_by, model.shared_with):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this batch job"
            )
        
        return BatchJobResponse.model_validate(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting batch job: {str(e)}"
        )


@router.post("/{job_id}/cancel", response_model=BatchJobResponse)
async def cancel_batch_job(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Cancel a batch prediction job.
    """
    try:
        job = await BatchPredictionJobService.get_batch_job(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch prediction job {job_id} not found"
            )
        
        # Check access
        model = await db.get(MLModel, job.model_id)
        if model and not has_resource_access(current_user, model.created_by, model.shared_with):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this batch job"
            )
        
        cancelled = await BatchPredictionJobService.cancel_job(db, job_id)
        
        if not cancelled:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be cancelled (already completed, failed, or cancelled)"
            )
        
        # Refresh job
        job = await BatchPredictionJobService.get_batch_job(db, job_id)
        return BatchJobResponse.model_validate(job)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error cancelling batch job {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error cancelling batch job: {str(e)}"
        )


@router.get("/{job_id}/results")
async def get_batch_job_results(
    job_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Download batch prediction job results.
    """
    try:
        job = await BatchPredictionJobService.get_batch_job(db, job_id)
        
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Batch prediction job {job_id} not found"
            )
        
        # Check access
        model = await db.get(MLModel, job.model_id)
        if model and not has_resource_access(current_user, model.created_by, model.shared_with):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this batch job"
            )
        
        if job.status != BatchJobStatus.COMPLETED:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Job is not completed (status: {job.status})"
            )
        
        if not job.result_path:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Result file not found"
            )
        
        # Download from cloud storage
        from app.storage.cloud_storage import cloud_storage
        from fastapi.responses import Response
        
        try:
            file_content = await cloud_storage.download_file(job.result_path)
            
            # Determine content type
            content_type = "text/csv"
            if job.result_format == "json":
                content_type = "application/json"
            elif job.result_format == "parquet":
                content_type = "application/octet-stream"
            
            return Response(
                content=file_content,
                media_type=content_type,
                headers={
                    "Content-Disposition": f'attachment; filename="batch_results_{job_id}.{job.result_format}"'
                }
            )
        except Exception as e:
            logger.error(f"Error downloading results for job {job_id}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error downloading results: {str(e)}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting batch job results {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting batch job results: {str(e)}"
        )


"""
Model retraining API endpoints.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.models.database.ml_models import MLModel
from app.models.schemas.ml import RetrainingScheduleRequest, RetrainingScheduleResponse
from app.pipelines.retraining_pipeline import automated_retraining_flow

router = APIRouter(prefix="/retrain", tags=["Model Retraining"])


@router.post("/schedule", response_model=RetrainingScheduleResponse)
async def schedule_retraining(
    request: RetrainingScheduleRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db)
):
    """Schedule model retraining."""
    # Verify model exists
    model = await db.get(MLModel, request.model_id)
    if not model:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found"
        )
    
    # Schedule retraining
    retraining_config = {
        "new_data_threshold": request.new_data_threshold,
        "performance_threshold": request.performance_threshold,
        "training_config": request.training_config or {}
    }
    
    # Run in background
    background_tasks.add_task(
        automated_retraining_flow,
        model_id=request.model_id,
        retraining_config=retraining_config
    )
    
    return RetrainingScheduleResponse(
        model_id=request.model_id,
        scheduled=True,
        message="Retraining scheduled successfully"
    )


@router.get("/history")
async def get_retraining_history(
    model_id: Optional[int] = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db)
):
    """Get retraining history."""
    from sqlalchemy import select, and_
    from app.models.database.training_jobs import TrainingJob
    
    query = select(TrainingJob).where(
        TrainingJob.model_id.isnot(None)
    )
    
    if model_id:
        query = query.where(TrainingJob.model_id == model_id)
    
    query = query.order_by(TrainingJob.created_at.desc()).limit(limit).offset(offset)
    
    result = await db.execute(query)
    jobs = result.scalars().all()
    
    return {
        "jobs": [
            {
                "id": job.id,
                "model_id": job.model_id,
                "status": job.status.value,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "metrics": job.metrics
            }
            for job in jobs
        ],
        "total": len(jobs)
    }


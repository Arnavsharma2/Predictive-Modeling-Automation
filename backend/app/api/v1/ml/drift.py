"""
Drift detection API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.ml_models import MLModel
from app.models.database.data_sources import DataSource
from app.models.database.data_points import DataPoint
from app.models.database.users import User
from app.middleware.auth import get_current_user
from app.core.permissions import has_resource_access
from typing import Annotated
from app.models.schemas.drift import (
    DriftCheckRequest,
    DriftReportResponse,
    DriftReportListResponse
)
from app.services.drift_monitoring import DriftMonitoringService
import pandas as pd

logger = get_logger(__name__)

router = APIRouter(prefix="/drift", tags=["Drift Detection"])


@router.post("/check", response_model=DriftReportResponse, status_code=status.HTTP_201_CREATED)
async def check_drift(
    request: DriftCheckRequest,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Check for data drift for a model.
    
    Requires current data to be provided via data_source_id or uploaded.
    """
    try:
        # Check model access
        model_result = await db.execute(
            select(MLModel).where(MLModel.id == request.model_id)
        )
        model = model_result.scalar_one_or_none()
        
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
        
        # Get current data
        current_data = None
        if request.current_data_source_id:
            # Use provided data source
            data_source_result = await db.execute(
                select(DataSource).where(DataSource.id == request.current_data_source_id)
            )
            data_source = data_source_result.scalar_one_or_none()
            
            if not data_source:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Data source {request.current_data_source_id} not found"
                )
            
            if not has_resource_access(current_user, data_source.created_by, data_source.shared_with):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="You don't have access to this data source"
                )
            
            # Get data points
            data_points_result = await db.execute(
                select(DataPoint)
                .where(DataPoint.source_id == data_source.id)
                .order_by(DataPoint.timestamp.desc())
                .limit(10000)
            )
            data_points = data_points_result.scalars().all()
            
            if data_points:
                data_list = []
                for point in data_points:
                    if isinstance(point.data, dict):
                        data_list.append(point.data)
                
                if data_list:
                    current_data = pd.DataFrame(data_list)
        else:
            # Use model's training data as current data
            if model.data_source_id:
                data_source_result = await db.execute(
                    select(DataSource).where(DataSource.id == model.data_source_id)
                )
                data_source = data_source_result.scalar_one_or_none()
                
                if data_source:
                    # Get data points from training data
                    data_points_result = await db.execute(
                        select(DataPoint)
                        .where(DataPoint.source_id == data_source.id)
                        .order_by(DataPoint.timestamp.desc())
                        .limit(10000)
                    )
                    data_points = data_points_result.scalars().all()
                    
                    if data_points:
                        data_list = []
                        for point in data_points:
                            if isinstance(point.data, dict):
                                data_list.append(point.data)
                        
                        if data_list:
                            current_data = pd.DataFrame(data_list)
        
        if current_data is None or len(current_data) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No current data available. Please provide a valid data_source_id with data points, or ensure the model has training data available."
            )
        
        # Check for drift
        monitoring_service = DriftMonitoringService()
        try:
            drift_report = await monitoring_service.check_drift(
                db=db,
                model_id=request.model_id,
                current_data=current_data,
                features=request.features,
                created_by=current_user.id if current_user else None
            )
        except HTTPException:
            # Re-raise HTTPException as-is (e.g., 400 for missing reference data)
            raise
        except ValueError as e:
            # Convert ValueError to HTTPException with 400 status
            if "No reference data" in str(e):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=str(e)
                )
            raise
        
        # Trigger retraining if needed
        if drift_report.severity.value == "high":
            await monitoring_service.trigger_retraining_if_needed(db, drift_report)
        
        return DriftReportResponse.from_orm(drift_report)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking drift: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking drift: {str(e)}"
        )


@router.get("/reports/{model_id}", response_model=DriftReportListResponse)
async def get_drift_reports(
    model_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get drift reports for a model."""
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
        
        if not has_resource_access(current_user, model.created_by, model.shared_with):
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
        from sqlalchemy import func
        from app.models.database.drift_reports import DriftReport
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


@router.get("/reports/{model_id}/latest", response_model=DriftReportResponse)
async def get_latest_drift_report(
    model_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get the latest drift report for a model."""
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
        
        if not has_resource_access(current_user, model.created_by, model.shared_with):
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


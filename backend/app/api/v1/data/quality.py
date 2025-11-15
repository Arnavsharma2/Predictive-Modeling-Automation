"""
Data quality API endpoints.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.data_sources import DataSource
from app.models.database.data_quality_reports import DataQualityReport
from app.models.database.users import User
from app.middleware.auth import get_current_user
from app.core.permissions import has_resource_access
from typing import Annotated
from app.models.schemas.data_quality import (
    DataQualityCheckRequest,
    DataQualityReportResponse,
    DataQualityReportListResponse,
    DataProfileResponse,
    DataLineageResponse
)
from app.services.data_quality import DataQualityService
from app.services.data_profiling import DataProfilingService
from app.services.data_lineage import DataLineageService

logger = get_logger(__name__)

router = APIRouter(prefix="/quality", tags=["Data Quality"])


@router.post("/check", response_model=DataQualityReportResponse, status_code=status.HTTP_201_CREATED)
async def check_data_quality(
    request: DataQualityCheckRequest,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """Check data quality for a data source."""
    try:
        # Check data source access
        data_source_result = await db.execute(
            select(DataSource).where(DataSource.id == request.data_source_id)
        )
        data_source = data_source_result.scalar_one_or_none()
        
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data source {request.data_source_id} not found"
            )
        
        if not has_resource_access(current_user, data_source):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this data source"
            )
        
        # Perform quality check
        quality_service = DataQualityService()
        quality_report = await quality_service.check_data_quality(
            db=db,
            data_source_id=request.data_source_id,
            created_by=current_user.id if current_user else None
        )
        
        # Update overall score
        if quality_report.quality_metrics and "summary" in quality_report.quality_metrics:
            quality_report.overall_score = quality_report.quality_metrics["summary"].get("overall_score")
            await db.commit()
            await db.refresh(quality_report)
        
        return DataQualityReportResponse.from_orm(quality_report)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking data quality: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error checking data quality: {str(e)}"
        )


@router.get("/{data_source_id}/reports", response_model=DataQualityReportListResponse)
async def get_quality_reports(
    data_source_id: int,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get quality reports for a data source."""
    try:
        # Check data source access
        data_source_result = await db.execute(
            select(DataSource).where(DataSource.id == data_source_id)
        )
        data_source = data_source_result.scalar_one_or_none()
        
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data source {data_source_id} not found"
            )
        
        if not has_resource_access(current_user, data_source):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this data source"
            )
        
        # Get reports
        reports_result = await db.execute(
            select(DataQualityReport)
            .where(DataQualityReport.data_source_id == data_source_id)
            .order_by(DataQualityReport.created_at.desc())
            .limit(limit)
            .offset(skip)
        )
        reports = list(reports_result.scalars().all())
        
        # Get total count
        count_result = await db.execute(
            select(func.count(DataQualityReport.id))
            .where(DataQualityReport.data_source_id == data_source_id)
        )
        total = count_result.scalar() or 0
        
        return DataQualityReportListResponse(
            reports=[DataQualityReportResponse.from_orm(r) for r in reports],
            total=total,
            limit=limit,
            offset=skip
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quality reports: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting quality reports: {str(e)}"
        )


@router.get("/{data_source_id}/profile", response_model=DataProfileResponse)
async def get_data_profile(
    data_source_id: int,
    sample_size: Optional[int] = Query(10000, ge=100, le=100000),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get data profile for a data source."""
    try:
        # Check data source access
        data_source_result = await db.execute(
            select(DataSource).where(DataSource.id == data_source_id)
        )
        data_source = data_source_result.scalar_one_or_none()
        
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data source {data_source_id} not found"
            )
        
        if not has_resource_access(current_user, data_source):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this data source"
            )
        
        # Get data
        quality_service = DataQualityService()
        data = await quality_service._get_data_from_source(db, data_source_id, limit=sample_size)
        
        if data is None or len(data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No data available for profiling"
            )
        
        # Generate profile
        profiling_service = DataProfilingService()
        profile = profiling_service.generate_profile(data, sample_size=sample_size)
        
        return DataProfileResponse(**profile)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating data profile: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating data profile: {str(e)}"
        )


@router.get("/{data_source_id}/lineage", response_model=List[DataLineageResponse])
async def get_data_lineage(
    data_source_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get data lineage for a data source."""
    try:
        # Check data source access
        data_source_result = await db.execute(
            select(DataSource).where(DataSource.id == data_source_id)
        )
        data_source = data_source_result.scalar_one_or_none()
        
        if not data_source:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Data source {data_source_id} not found"
            )
        
        if not has_resource_access(current_user, data_source):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have access to this data source"
            )
        
        # Get lineage
        lineage_service = DataLineageService()
        lineage_records = await lineage_service.get_lineage_for_source(db, data_source_id)
        
        return [DataLineageResponse.from_orm(l) for l in lineage_records]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting data lineage: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting data lineage: {str(e)}"
        )


"""
Data preview and exploration API endpoints.
"""
from typing import Optional, List
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from app.core.database import get_db
from app.core.logging import get_logger
from app.models.database.data_sources import DataSource
from app.models.database.data_points import DataPoint
from app.models.database.users import User
from app.middleware.auth import get_current_user
from app.core.permissions import has_resource_access, filter_by_user_access
from typing import Annotated

logger = get_logger(__name__)

# Handle Prefect TerminationSignal exceptions
try:
    from prefect.exceptions import TerminationSignal
except ImportError:
    TerminationSignal = None

router = APIRouter(prefix="/data", tags=["Data"])

# Include quality router
from .quality import router as quality_router
router.include_router(quality_router)


@router.get("/preview")
async def preview_data(
    source_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Preview ingested data for a data source.
    """
    try:
        # Verify source exists and user has access
        source = await db.get(DataSource, source_id)
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Data source {source_id} not found"
            )
        
        # Check user has access to data source
        if not has_resource_access(current_user, source.created_by, source.shared_with):
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this data source"
            )
        
        # Get data points
        result = await db.execute(
            select(DataPoint)
            .where(DataPoint.source_id == source_id)
            .order_by(DataPoint.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        data_points = result.scalars().all()
        
        # Get total count
        count_result = await db.execute(
            select(func.count(DataPoint.id))
            .where(DataPoint.source_id == source_id)
        )
        total = count_result.scalar() or 0
        
        return {
            "source_id": source_id,
            "source_name": source.name,
            "data": [
                {
                    "id": dp.id,
                    "data": dp.data,
                    "metadata": dp.meta_data,
                    "timestamp": dp.timestamp.isoformat() if dp.timestamp else None
                }
                for dp in data_points
            ],
            "pagination": {
                "offset": offset,
                "limit": limit,
                "total": total
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Handle Prefect TerminationSignal (occurs during uvicorn reload)
        if TerminationSignal and isinstance(e, TerminationSignal):
            logger.warning(f"Prefect TerminationSignal caught during data preview (likely due to reload): {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to reload. Please retry."
            )
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error previewing data: {str(e)}"
        )


@router.get("/sources")
async def list_data_sources_with_stats(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    List all data sources with statistics (filtered by user access).
    """
    try:
        # Get sources with user access filter
        query = select(DataSource)
        query = filter_by_user_access(query, DataSource, current_user)
        
        result = await db.execute(
            query
            .offset(skip)
            .limit(limit)
            .order_by(DataSource.created_at.desc())
        )
        sources = result.scalars().all()
        
        # Get source IDs
        source_ids = [source.id for source in sources]
        
        # Get statistics for all sources in a single query (optimized to avoid N+1)
        stats_map = {}
        if source_ids:
            stats_result = await db.execute(
                select(
                    DataPoint.source_id,
                    func.count(DataPoint.id).label('data_point_count'),
                    func.max(DataPoint.timestamp).label('latest_timestamp')
                )
                .where(DataPoint.source_id.in_(source_ids))
                .group_by(DataPoint.source_id)
            )
            stats_rows = stats_result.all()
            
            # Create a map of source_id -> stats
            for row in stats_rows:
                stats_map[row.source_id] = {
                    "data_point_count": row.data_point_count,
                    "latest_timestamp": row.latest_timestamp
                }
        
        # Build response with stats
        sources_with_stats = []
        for source in sources:
            stats = stats_map.get(source.id, {
                "data_point_count": 0,
                "latest_timestamp": None
            })
            
            sources_with_stats.append({
                "id": source.id,
                "name": source.name,
                "type": source.type.value,
                "status": source.status.value,
                "description": source.description,
                "created_at": source.created_at.isoformat() if source.created_at else None,
                "updated_at": source.updated_at.isoformat() if source.updated_at else None,
                "stats": {
                    "data_point_count": stats["data_point_count"],
                    "latest_timestamp": stats["latest_timestamp"].isoformat() if stats["latest_timestamp"] else None
                }
            })
        
        # Get total count
        total_result = await db.execute(select(func.count(DataSource.id)))
        total = total_result.scalar() or 0
        
        return {
            "sources": sources_with_stats,
            "pagination": {
                "skip": skip,
                "limit": limit,
                "total": total
            }
        }
        
    except Exception as e:
        # Handle Prefect TerminationSignal (occurs during uvicorn reload)
        if TerminationSignal and isinstance(e, TerminationSignal):
            logger.warning(f"Prefect TerminationSignal caught during data listing (likely due to reload): {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to reload. Please retry."
            )
        logger.error(f"Error listing data sources with stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error listing data sources: {str(e)}"
        )


@router.get("/stats")
async def get_data_statistics(
    source_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics for a data source.
    """
    try:
        # Verify source exists and user has access
        source = await db.get(DataSource, source_id)
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Data source {source_id} not found"
            )
        
        # Check user has access to data source
        if not has_resource_access(current_user, source.created_by, source.shared_with):
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this data source"
            )
        
        # Get total count
        count_result = await db.execute(
            select(func.count(DataPoint.id))
            .where(DataPoint.source_id == source_id)
        )
        total_count = count_result.scalar() or 0
        
        # Get date range
        date_range_result = await db.execute(
            select(
                func.min(DataPoint.timestamp),
                func.max(DataPoint.timestamp)
            )
            .where(DataPoint.source_id == source_id)
        )
        date_range = date_range_result.first()
        min_timestamp = date_range[0] if date_range else None
        max_timestamp = date_range[1] if date_range else None
        
        # Get sample data to determine columns
        sample_result = await db.execute(
            select(DataPoint)
            .where(DataPoint.source_id == source_id)
            .limit(1)
        )
        sample_point = sample_result.scalar_one_or_none()
        
        columns = []
        if sample_point and sample_point.data:
            # Extract column names from the first data point
            columns = list(sample_point.data.keys())
        
        # Get count by day (for time-series visualization)
        # This is a simplified version - in production, you might want more detailed stats
        daily_counts = []
        if min_timestamp and max_timestamp:
            # Get approximate daily counts
            # Note: This is a simplified query - for production, use TimescaleDB time_bucket
            pass
        
        # Return structure that matches frontend expectations
        return {
            "source_id": source_id,
            "source_name": source.name,
            "total_records": total_count,  # Frontend expects this
            "columns": columns,  # Frontend expects this
            "statistics": {
                "total_data_points": total_count,
                "date_range": {
                    "min": min_timestamp.isoformat() if min_timestamp else None,
                    "max": max_timestamp.isoformat() if max_timestamp else None
                },
                "daily_counts": daily_counts  # Can be enhanced with TimescaleDB queries
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Handle Prefect TerminationSignal (occurs during uvicorn reload)
        if TerminationSignal and isinstance(e, TerminationSignal):
            logger.warning(f"Prefect TerminationSignal caught during data statistics (likely due to reload): {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to reload. Please retry."
            )
        logger.error(f"Error getting data statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting data statistics: {str(e)}"
        )


@router.get("/sources/{source_id}/preview")
async def preview_data_by_source_id(
    source_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Preview ingested data for a data source (by source ID in path).
    """
    try:
        # Verify source exists and user has access
        source = await db.get(DataSource, source_id)
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Data source {source_id} not found"
            )
        
        # Check user has access to data source
        if not has_resource_access(current_user, source.created_by, source.shared_with):
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this data source"
            )
        
        # Get data points
        result = await db.execute(
            select(DataPoint)
            .where(DataPoint.source_id == source_id)
            .order_by(DataPoint.timestamp.desc())
            .offset(offset)
            .limit(limit)
        )
        data_points = result.scalars().all()
        
        # Get total count
        count_result = await db.execute(
            select(func.count(DataPoint.id))
            .where(DataPoint.source_id == source_id)
        )
        total = count_result.scalar() or 0
        
        return {
            "source_id": source_id,
            "source_name": source.name,
            "data": [
                {
                    "id": dp.id,
                    "data": dp.data,
                    "metadata": dp.meta_data,
                    "timestamp": dp.timestamp.isoformat() if dp.timestamp else None
                }
                for dp in data_points
            ],
            "pagination": {
                "offset": offset,
                "limit": limit,
                "total": total
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Handle Prefect TerminationSignal (occurs during uvicorn reload)
        if TerminationSignal and isinstance(e, TerminationSignal):
            logger.warning(f"Prefect TerminationSignal caught during data preview (likely due to reload): {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to reload. Please retry."
            )
        logger.error(f"Error previewing data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error previewing data: {str(e)}"
        )


@router.get("/sources/{source_id}/stats")
async def get_data_statistics_by_source_id(
    source_id: int,
    current_user: Annotated[User, Depends(get_current_user)] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Get statistics for a data source (by source ID in path).
    """
    try:
        # Verify source exists and user has access
        source = await db.get(DataSource, source_id)
        if not source:
            raise HTTPException(
                status_code=404,
                detail=f"Data source {source_id} not found"
            )
        
        # Check user has access to data source
        if not has_resource_access(current_user, source.created_by, source.shared_with):
            raise HTTPException(
                status_code=403,
                detail="You don't have access to this data source"
            )
        
        # Get total count
        count_result = await db.execute(
            select(func.count(DataPoint.id))
            .where(DataPoint.source_id == source_id)
        )
        total_count = count_result.scalar() or 0
        
        # Get date range
        date_range_result = await db.execute(
            select(
                func.min(DataPoint.timestamp),
                func.max(DataPoint.timestamp)
            )
            .where(DataPoint.source_id == source_id)
        )
        date_range = date_range_result.first()
        min_timestamp = date_range[0] if date_range else None
        max_timestamp = date_range[1] if date_range else None
        
        # Get sample data to determine columns
        sample_result = await db.execute(
            select(DataPoint)
            .where(DataPoint.source_id == source_id)
            .limit(1)
        )
        sample_point = sample_result.scalar_one_or_none()
        
        columns = []
        if sample_point and sample_point.data:
            # Extract column names from the first data point
            columns = list(sample_point.data.keys())
        
        # Get count by day (for time-series visualization)
        # This is a simplified version - in production, you might want more detailed stats
        daily_counts = []
        if min_timestamp and max_timestamp:
            # Get approximate daily counts
            # Note: This is a simplified query - for production, use TimescaleDB time_bucket
            pass
        
        # Return structure that matches frontend expectations
        return {
            "source_id": source_id,
            "source_name": source.name,
            "total_records": total_count,  # Frontend expects this
            "columns": columns,  # Frontend expects this
            "statistics": {
                "total_data_points": total_count,
                "date_range": {
                    "min": min_timestamp.isoformat() if min_timestamp else None,
                    "max": max_timestamp.isoformat() if max_timestamp else None
                },
                "daily_counts": daily_counts  # Can be enhanced with TimescaleDB queries
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        # Handle Prefect TerminationSignal (occurs during uvicorn reload)
        if TerminationSignal and isinstance(e, TerminationSignal):
            logger.warning(f"Prefect TerminationSignal caught during data statistics (likely due to reload): {e}")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable due to reload. Please retry."
            )
        logger.error(f"Error getting data statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting data statistics: {str(e)}"
        )


@router.delete("/points/{point_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_data_point(
    point_id: int,
    db: AsyncSession = Depends(get_db)
):
    """
    Delete a single data point.
    """
    try:
        point = await db.get(DataPoint, point_id)
        if not point:
            raise HTTPException(
                status_code=404,
                detail=f"Data point {point_id} not found"
            )
        
        await db.delete(point)
        await db.commit()
        
        logger.info(f"Successfully deleted data point {point_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting data point {point_id}: {e}", exc_info=True)
        await db.rollback()
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting data point: {str(e)}"
        )


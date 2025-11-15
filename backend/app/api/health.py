"""
Health check endpoint.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from typing import Dict, Any

from app.core.database import get_db
from app.core.config import settings

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    """
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/health/detailed")
async def detailed_health_check(
    db: AsyncSession = Depends(get_db)
) -> Dict[str, Any]:
    """
    Detailed health check including database connectivity.
    """
    health_status = {
        "status": "healthy",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "checks": {
            "database": "unknown",
        },
    }
    
    # Check database connection
    try:
        result = await db.execute(text("SELECT 1"))
        result.scalar()
        health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["status"] = "unhealthy"
        health_status["checks"]["database"] = f"unhealthy: {str(e)}"
    
    return health_status


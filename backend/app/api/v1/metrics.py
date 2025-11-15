"""
Metrics endpoint for Prometheus scraping.
"""
from fastapi import APIRouter, Response
from app.core.metrics import get_metrics, get_metrics_content_type

router = APIRouter(tags=["Metrics"])


@router.get("/metrics")
async def prometheus_metrics():
    """
    Expose Prometheus metrics for scraping.

    Returns:
        Prometheus metrics in text format
    """
    metrics = get_metrics()
    return Response(
        content=metrics,
        media_type=get_metrics_content_type()
    )

"""
Metrics collection middleware for Prometheus.
"""
import time
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from app.core.metrics import (
    http_requests_total,
    http_request_duration_seconds,
    http_requests_in_progress
)


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect HTTP metrics for Prometheus."""

    def __init__(self, app: ASGIApp):
        """
        Initialize metrics middleware.

        Args:
            app: ASGI application
        """
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics.

        Args:
            request: HTTP request
            call_next: Next middleware/handler

        Returns:
            HTTP response
        """
        # Extract method and path
        method = request.method
        path = request.url.path

        # Normalize path to avoid high cardinality
        # Group dynamic paths (e.g., /api/v1/models/123 -> /api/v1/models/{id})
        endpoint = self._normalize_path(path)

        # Track in-progress requests
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()

        # Start timer
        start_time = time.time()

        try:
            # Process request
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Record metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=response.status_code
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            return response

        except Exception as e:
            # Calculate duration even on error
            duration = time.time() - start_time

            # Record error metrics
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=500
            ).inc()

            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)

            raise e

        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()

    def _normalize_path(self, path: str) -> str:
        """
        Normalize path to reduce cardinality.

        Args:
            path: Original request path

        Returns:
            Normalized path with IDs replaced by placeholders
        """
        parts = path.split('/')

        # Replace numeric IDs and UUIDs with placeholders
        normalized_parts = []
        for part in parts:
            if part.isdigit():
                normalized_parts.append('{id}')
            elif len(part) == 36 and part.count('-') == 4:  # UUID pattern
                normalized_parts.append('{uuid}')
            else:
                normalized_parts.append(part)

        return '/'.join(normalized_parts)

"""
Request/response logging middleware.
"""
from typing import Callable
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import time
import json

from app.core.logging import get_logger

logger = get_logger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Request/response logging middleware."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        start_time = time.time()
        
        # Log request
        request_log = {
            "method": request.method,
            "path": request.url.path,
            "query_params": dict(request.query_params),
            "client_host": request.client.host if request.client else None,
            "user_agent": request.headers.get("user-agent"),
        }
        
        logger.info(f"Request: {request.method} {request.url.path}", extra={"request": request_log})
        
        # Process request
        response = await call_next(request)
        
        # Calculate duration
        duration_ms = (time.time() - start_time) * 1000
        
        # Log response
        response_log = {
            "status_code": response.status_code,
            "duration_ms": duration_ms,
        }
        
        log_level = "error" if response.status_code >= 500 else "warning" if response.status_code >= 400 else "info"
        getattr(logger, log_level)(
            f"Response: {request.method} {request.url.path} - {response.status_code} ({duration_ms:.2f}ms)",
            extra={"response": response_log}
        )
        
        return response


"""
Rate limiting middleware.
"""
from typing import Callable
from fastapi import Request, Response, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
import time

from app.cache.redis_client import redis_client
from app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware."""
    
    def __init__(
        self,
        app,
        default_limit: int = 100,
        default_window: int = 60,  # seconds
        per_endpoint_limits: dict = None,
        per_user_limits: dict = None
    ):
        """
        Initialize rate limiting middleware.
        
        Args:
            app: FastAPI application
            default_limit: Default requests per window
            default_window: Default time window in seconds
            per_endpoint_limits: Dict mapping endpoint paths to limits
            per_user_limits: Dict mapping user IDs to limits
        """
        super().__init__(app)
        self.default_limit = default_limit
        self.default_window = default_window
        self.per_endpoint_limits = per_endpoint_limits or {}
        self.per_user_limits = per_user_limits or {}
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request with rate limiting.
        
        Args:
            request: FastAPI request
            call_next: Next middleware/handler
            
        Returns:
            Response
        """
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Get endpoint path
        endpoint = request.url.path
        
        # Determine rate limit
        limit, window = self._get_rate_limit(endpoint, client_id)
        
        # Check rate limit
        key = f"rate_limit:{endpoint}:{client_id}"
        
        # Get current count
        current_count = await redis_client.get(key)
        if current_count is None:
            current_count = 0
        else:
            current_count = int(current_count)
        
        # Check if limit exceeded
        if current_count >= limit:
            logger.warning(f"Rate limit exceeded for {client_id} on {endpoint}")
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit} requests per {window} seconds",
                headers={
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(time.time()) + window)
                }
            )
        
        # Increment counter
        await redis_client.set(key, current_count + 1, ttl=window)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = limit - (current_count + 1)
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(time.time()) + window)
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier
        """
        # Try to get user ID from headers or auth
        user_id = request.headers.get("X-User-ID")
        if user_id:
            return f"user:{user_id}"
        
        # Fallback to IP address
        client_host = request.client.host if request.client else "unknown"
        return f"ip:{client_host}"
    
    def _get_rate_limit(self, endpoint: str, client_id: str) -> tuple[int, int]:
        """
        Get rate limit for endpoint and client.
        
        Args:
            endpoint: Endpoint path
            client_id: Client identifier
            
        Returns:
            Tuple of (limit, window)
        """
        # Check per-user limits
        if client_id.startswith("user:") and self.per_user_limits:
            user_id = client_id.replace("user:", "")
            if user_id in self.per_user_limits:
                return self.per_user_limits[user_id]
        
        # Check per-endpoint limits (check longer/more specific patterns first)
        # Sort by length descending to match more specific patterns before general ones
        sorted_patterns = sorted(
            self.per_endpoint_limits.items(),
            key=lambda x: len(x[0]),
            reverse=True
        )
        for path_pattern, (limit, window) in sorted_patterns:
            # Match if endpoint starts with pattern (allows for path parameters)
            if endpoint.startswith(path_pattern):
                return (limit, window)
        
        # Default limit
        return (self.default_limit, self.default_window)


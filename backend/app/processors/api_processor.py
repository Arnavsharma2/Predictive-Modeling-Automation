"""
API data processor for external API integration.
"""
import asyncio
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone

import httpx
from app.core.logging import get_logger
from app.models.schemas.ingestion import APIAuthConfig

logger = get_logger(__name__)


class APIProcessor:
    """Processor for API data sources."""
    
    DEFAULT_TIMEOUT = 30.0  # seconds
    DEFAULT_RETRY_DELAY = 1  # seconds
    
    def __init__(self):
        self.logger = logger
        self._rate_limit_tracker: Dict[str, List[float]] = {}  # Track request timestamps per endpoint
    
    def _prepare_auth(self, auth: Optional[APIAuthConfig]) -> Dict[str, str]:
        """
        Prepare authentication headers based on auth config.
        
        Args:
            auth: Authentication configuration
            
        Returns:
            Dictionary with authentication headers
        """
        if not auth:
            return {}
        
        headers = {}
        
        if auth.auth_type == "api_key":
            if auth.api_key:
                header_name = auth.api_key_header or "X-API-Key"
                headers[header_name] = auth.api_key
        
        elif auth.auth_type == "bearer":
            if auth.bearer_token:
                headers["Authorization"] = f"Bearer {auth.bearer_token}"
        
        elif auth.auth_type == "basic":
            if auth.username and auth.password:
                import base64
                credentials = base64.b64encode(
                    f"{auth.username}:{auth.password}".encode()
                ).decode()
                headers["Authorization"] = f"Basic {credentials}"
        
        # OAuth would require token exchange, which is more complex
        # For now, we'll just use bearer token if provided
        elif auth.auth_type == "oauth":
            if auth.bearer_token:
                headers["Authorization"] = f"Bearer {auth.bearer_token}"
            # TODO: Implement full OAuth flow if needed
        
        return headers
    
    async def _check_rate_limit(self, url: str, rate_limit: Optional[int]) -> None:
        """
        Check and enforce rate limiting.
        
        Args:
            url: API endpoint URL
            rate_limit: Maximum requests per minute
        """
        if not rate_limit:
            return
        
        now = time.time()
        minute_ago = now - 60
        
        # Clean old timestamps
        if url in self._rate_limit_tracker:
            self._rate_limit_tracker[url] = [
                ts for ts in self._rate_limit_tracker[url] if ts > minute_ago
            ]
        else:
            self._rate_limit_tracker[url] = []
        
        # Check if we've exceeded rate limit
        if len(self._rate_limit_tracker[url]) >= rate_limit:
            # Calculate wait time
            oldest_request = min(self._rate_limit_tracker[url])
            wait_time = 60 - (now - oldest_request) + 1
            if wait_time > 0:
                logger.info(f"Rate limit reached for {url}. Waiting {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
        
        # Record this request
        self._rate_limit_tracker[url].append(now)
    
    async def fetch_data(
        self,
        url: str,
        method: str = "GET",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        auth: Optional[APIAuthConfig] = None,
        rate_limit: Optional[int] = None,
        retry_count: int = 3,
        retry_delay: int = 1,
        timeout: float = DEFAULT_TIMEOUT
    ) -> Dict[str, Any]:
        """
        Fetch data from API with retry logic and rate limiting.
        
        Args:
            url: API endpoint URL
            method: HTTP method
            headers: Additional headers
            params: Query parameters
            body: Request body (for POST/PUT)
            auth: Authentication configuration
            rate_limit: Rate limit (requests per minute)
            retry_count: Number of retry attempts
            retry_delay: Delay between retries (seconds)
            timeout: Request timeout (seconds)
            
        Returns:
            Dictionary with response data and metadata
        """
        # Prepare headers
        request_headers = headers or {}
        auth_headers = self._prepare_auth(auth)
        request_headers.update(auth_headers)
        
        # Check rate limit
        await self._check_rate_limit(url, rate_limit)
        
        last_exception = None
        
        for attempt in range(retry_count + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    if method.upper() == "GET":
                        response = await client.get(url, headers=request_headers, params=params)
                    elif method.upper() == "POST":
                        response = await client.post(url, headers=request_headers, params=params, json=body)
                    elif method.upper() == "PUT":
                        response = await client.put(url, headers=request_headers, params=params, json=body)
                    elif method.upper() == "PATCH":
                        response = await client.patch(url, headers=request_headers, params=params, json=body)
                    elif method.upper() == "DELETE":
                        response = await client.delete(url, headers=request_headers, params=params)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    response.raise_for_status()
                    
                    # Try to parse as JSON
                    try:
                        data = response.json()
                    except Exception:
                        data = {"raw": response.text}
                    
                    return {
                        "data": data,
                        "status_code": response.status_code,
                        "headers": dict(response.headers),
                        "metadata": {
                            "url": str(response.url),
                            "method": method,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "attempt": attempt + 1
                        }
                    }
                    
            except httpx.HTTPStatusError as e:
                # Don't retry on client errors (4xx)
                if 400 <= e.response.status_code < 500:
                    logger.error(f"Client error fetching {url}: {e}")
                    raise
                last_exception = e
                logger.warning(f"HTTP error fetching {url} (attempt {attempt + 1}/{retry_count + 1}): {e}")
                
            except (httpx.RequestError, httpx.TimeoutException) as e:
                last_exception = e
                logger.warning(f"Error fetching {url} (attempt {attempt + 1}/{retry_count + 1}): {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < retry_count:
                await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        logger.error(f"Failed to fetch {url} after {retry_count + 1} attempts")
        raise Exception(f"Failed to fetch data from {url} after {retry_count + 1} attempts: {last_exception}")


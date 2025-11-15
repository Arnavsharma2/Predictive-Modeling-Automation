"""
Caching decorators for API endpoints and functions.
"""
from functools import wraps
from typing import Callable, Any, Optional
import hashlib
import json

from app.cache.redis_client import redis_client
from app.core.logging import get_logger

logger = get_logger(__name__)


def cache_key(*args, **kwargs) -> str:
    """
    Generate cache key from function arguments.
    
    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Cache key string
    """
    # Create a hash of the arguments
    key_data = {
        "args": args,
        "kwargs": kwargs
    }
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()
    return key_hash


def cached(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    exclude_args: Optional[list] = None
):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        exclude_args: List of argument names to exclude from cache key
        
    Usage:
        @cached(ttl=3600, key_prefix="model")
        async def get_model(model_id: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Build cache key
            cache_kwargs = kwargs.copy()
            if exclude_args:
                for arg in exclude_args:
                    cache_kwargs.pop(arg, None)
            
            key = f"{key_prefix}:{func.__name__}:{cache_key(*args, **cache_kwargs)}"
            
            # Try to get from cache
            cached_value = await redis_client.get(key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {key}")
                return cached_value
            
            # Execute function
            logger.debug(f"Cache miss for {key}")
            result = await func(*args, **kwargs)
            
            # Store in cache
            await redis_client.set(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(pattern: str):
    """
    Decorator to invalidate cache after function execution.
    
    Args:
        pattern: Cache key pattern to invalidate
        
    Usage:
        @invalidate_cache("model:*")
        async def update_model(model_id: int):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            await redis_client.clear_pattern(pattern)
            logger.debug(f"Invalidated cache pattern: {pattern}")
            
            return result
        
        return wrapper
    return decorator


"""
Redis client for caching.
"""
import json
from typing import Optional, Any
import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Async Redis client for caching."""
    
    def __init__(self):
        """Initialize Redis client."""
        self.client: Optional[Redis] = None
        self._connect()
    
    def _connect(self):
        """Connect to Redis."""
        try:
            self.client = redis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True
            )
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.client:
            return None
        
        try:
            value = await self.client.get(key)
            if value:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    return value
            return None
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (uses default if None)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            if isinstance(value, (dict, list)):
                value = json.dumps(value)
            
            ttl = ttl or settings.REDIS_CACHE_TTL
            await self.client.setex(key, ttl, value)
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if successful, False otherwise
        """
        if not self.client:
            return False
        
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists, False otherwise
        """
        if not self.client:
            return False
        
        try:
            return await self.client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """
        Clear all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "model:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.client:
            return 0
        
        try:
            keys = []
            async for key in self.client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Error clearing cache pattern: {e}")
            return 0
    
    async def close(self):
        """Close Redis connection."""
        if self.client:
            await self.client.close()
            self.client = None


# Global instance
redis_client = RedisClient()


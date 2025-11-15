"""
Redis Pub/Sub for cross-process communication.

This allows Celery workers to send WebSocket updates to the main FastAPI process.
"""
import json
import redis
import asyncio
from typing import Dict, Any, Optional
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RedisPubSub:
    """Redis Pub/Sub client for cross-process communication."""

    def __init__(self):
        """Initialize Redis connection."""
        # Parse Redis URL
        redis_url = settings.REDIS_URL
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.pubsub = None
        self.listener_task = None

    def publish_training_update(self, job_id: int, data: Dict[str, Any]):
        """
        Publish a training progress update.

        Args:
            job_id: Training job ID
            data: Update data
        """
        channel = f"training_{job_id}"
        message = json.dumps(data)
        try:
            self.redis_client.publish(channel, message)
            logger.debug(f"Published update to {channel}: {data.get('status', 'unknown')}")
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")

    def publish_batch_update(self, job_id: int, data: Dict[str, Any]):
        """
        Publish a batch prediction progress update.

        Args:
            job_id: Batch job ID
            data: Update data
        """
        channel = f"batch_{job_id}"
        message = json.dumps(data)
        try:
            self.redis_client.publish(channel, message)
            logger.debug(f"Published update to {channel}")
        except Exception as e:
            logger.error(f"Error publishing to Redis: {e}")

    async def subscribe_and_forward(self, ws_manager):
        """
        Subscribe to Redis channels and forward messages to WebSocket manager.

        Args:
            ws_manager: WebSocket connection manager
        """
        # Create a new Redis connection for pub/sub
        # (pub/sub connections can't be used for other commands)
        pubsub_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
        self.pubsub = pubsub_client.pubsub()

        # Subscribe to all training and batch channels
        # Using pattern subscription to catch all channels
        self.pubsub.psubscribe("training_*", "batch_*")

        logger.info("Redis Pub/Sub listener subscribed to patterns: training_*, batch_*")

        try:
            while True:
                # Check for messages
                message = self.pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)

                if message and message['type'] == 'pmessage':
                    channel = message['channel']
                    data = json.loads(message['data'])

                    logger.debug(f"Received Redis pub/sub message on {channel}: {data.get('status', 'unknown')}")

                    # Forward to WebSocket manager
                    await ws_manager.broadcast_job_update(channel, data)

                # Yield control to event loop
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info("Redis Pub/Sub listener cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in Redis Pub/Sub listener: {e}", exc_info=True)
        finally:
            self.pubsub.close()
            pubsub_client.close()

    def close(self):
        """Close Redis connection."""
        if self.pubsub:
            self.pubsub.close()
        if self.redis_client:
            self.redis_client.close()


# Global instance
redis_pubsub = RedisPubSub()

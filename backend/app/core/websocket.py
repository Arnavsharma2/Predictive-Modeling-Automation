"""
WebSocket connection manager for real-time updates.
"""
from typing import Dict, Set, List, Optional
from fastapi import WebSocket
import json
import asyncio
from app.core.logging import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""

    def __init__(self):
        """Initialize connection manager."""
        # Dictionary mapping subscription_key to set of connected websockets
        # subscription_key format: "{type}_{id}" (e.g., "training_123", "batch_456", "model_789", "dashboard_user_1")
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Dictionary mapping websocket to user_id for authentication
        self.connection_users: Dict[WebSocket, int] = {}
        # Dictionary mapping websocket to list of subscriptions
        self.websocket_subscriptions: Dict[WebSocket, Set[str]] = {}
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket, subscription_key: str, user_id: int, accept: bool = True):
        """
        Register a new WebSocket connection for a specific subscription.

        Args:
            websocket: WebSocket connection
            subscription_key: Subscription key (e.g., "training_123", "batch_456", "model_789", "dashboard_user_1")
            user_id: User ID for authentication
            accept: Whether to accept the websocket connection (default True, set False if already accepted)
        """
        if accept:
            await websocket.accept()

        async with self._lock:
            if subscription_key not in self.active_connections:
                self.active_connections[subscription_key] = set()

            self.active_connections[subscription_key].add(websocket)
            self.connection_users[websocket] = user_id
            
            # Track subscriptions for this websocket
            if websocket not in self.websocket_subscriptions:
                self.websocket_subscriptions[websocket] = set()
            self.websocket_subscriptions[websocket].add(subscription_key)

        logger.info(f"WebSocket connected for subscription {subscription_key} by user {user_id}")
    
    async def subscribe(self, websocket: WebSocket, subscription_key: str, user_id: int):
        """
        Subscribe to an additional channel on an existing connection.

        Args:
            websocket: Existing WebSocket connection
            subscription_key: Subscription key to add
            user_id: User ID for authentication
        """
        async with self._lock:
            if subscription_key not in self.active_connections:
                self.active_connections[subscription_key] = set()

            self.active_connections[subscription_key].add(websocket)
            
            # Track subscriptions for this websocket
            if websocket not in self.websocket_subscriptions:
                self.websocket_subscriptions[websocket] = set()
            self.websocket_subscriptions[websocket].add(subscription_key)

        logger.info(f"WebSocket subscribed to {subscription_key} by user {user_id}")
    
    async def unsubscribe(self, websocket: WebSocket, subscription_key: str):
        """
        Unsubscribe from a channel on an existing connection.

        Args:
            websocket: Existing WebSocket connection
            subscription_key: Subscription key to remove
        """
        async with self._lock:
            if subscription_key in self.active_connections:
                self.active_connections[subscription_key].discard(websocket)
                
                # Clean up empty subscriptions
                if not self.active_connections[subscription_key]:
                    del self.active_connections[subscription_key]
            
            # Remove from websocket subscriptions
            if websocket in self.websocket_subscriptions:
                self.websocket_subscriptions[websocket].discard(subscription_key)

        logger.info(f"WebSocket unsubscribed from {subscription_key}")

    async def disconnect(self, websocket: WebSocket, subscription_key: Optional[str] = None):
        """
        Unregister a WebSocket connection.

        Args:
            websocket: WebSocket connection to disconnect
            subscription_key: Optional specific subscription to disconnect from (None = disconnect from all)
        """
        async with self._lock:
            # Get all subscriptions for this websocket
            subscriptions = self.websocket_subscriptions.get(websocket, set())
            
            if subscription_key:
                # Disconnect from specific subscription
                subscriptions = {subscription_key}
            
            # Remove from all subscriptions
            for sub_key in subscriptions:
                if sub_key in self.active_connections:
                    self.active_connections[sub_key].discard(websocket)
                    
                    # Clean up empty subscriptions
                    if not self.active_connections[sub_key]:
                        del self.active_connections[sub_key]
            
            # Remove user mapping and subscription tracking
            if websocket in self.connection_users:
                user_id = self.connection_users[websocket]
                del self.connection_users[websocket]
                
                if subscription_key:
                    logger.info(f"WebSocket disconnected from {subscription_key} by user {user_id}")
                else:
                    logger.info(f"WebSocket fully disconnected by user {user_id}")
            
            if websocket in self.websocket_subscriptions:
                if subscription_key:
                    self.websocket_subscriptions[websocket].discard(subscription_key)
                else:
                    del self.websocket_subscriptions[websocket]

    async def broadcast_job_update(self, subscription_key: str, data: dict):
        """
        Broadcast an update to all clients subscribed to a specific channel.

        Args:
            subscription_key: Subscription key (e.g., "training_123", "batch_456", "model_789")
            data: Update data to send (will be JSON serialized)
        """
        if subscription_key not in self.active_connections:
            logger.debug(f"No active WebSocket connections for {subscription_key}")
            return

        connection_count = len(self.active_connections[subscription_key])
        logger.debug(f"Broadcasting to {connection_count} connection(s) on {subscription_key}: {data.get('status', 'unknown')}")

        message = json.dumps(data)
        disconnected = set()

        for websocket in self.active_connections[subscription_key].copy():
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                disconnected.add(websocket)

        # Clean up disconnected websockets
        if disconnected:
            async with self._lock:
                for websocket in disconnected:
                    if subscription_key in self.active_connections:
                        self.active_connections[subscription_key].discard(websocket)
                    if websocket in self.connection_users:
                        del self.connection_users[websocket]
                    if websocket in self.websocket_subscriptions:
                        self.websocket_subscriptions[websocket].discard(subscription_key)
    
    async def broadcast_prediction(self, model_id: int, prediction_data: dict):
        """
        Broadcast a prediction result to all clients subscribed to a model.

        Args:
            model_id: Model ID
            prediction_data: Prediction data to send
        """
        subscription_key = f"model_{model_id}"
        await self.broadcast_job_update(subscription_key, {
            "type": "prediction",
            "model_id": model_id,
            **prediction_data
        })
    
    async def broadcast_dashboard_update(self, user_id: int, update_data: dict):
        """
        Broadcast a dashboard update to a specific user.

        Args:
            user_id: User ID
            update_data: Update data to send
        """
        subscription_key = f"dashboard_user_{user_id}"
        await self.broadcast_job_update(subscription_key, {
            "type": "dashboard_update",
            "user_id": user_id,
            **update_data
        })

    async def send_personal_message(self, message: str, websocket: WebSocket):
        """
        Send a message to a specific WebSocket connection.

        Args:
            message: Message to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_text(message)
        except Exception as e:
            error_str = str(e)
            # Normal disconnection errors (1001 going away, no close frame) are expected
            # when clients disconnect and don't need to be logged as errors
            if "1001" in error_str or "going away" in error_str or "no close frame" in error_str:
                logger.debug(f"WebSocket disconnected (normal): {error_str}")
            else:
                logger.error(f"Error sending personal message: {e}")

    def get_connection_count(self, subscription_key: str) -> int:
        """
        Get the number of active connections for a subscription.

        Args:
            subscription_key: Subscription key

        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(subscription_key, set()))
    
    def get_websocket_subscriptions(self, websocket: WebSocket) -> Set[str]:
        """
        Get all subscriptions for a websocket.

        Args:
            websocket: WebSocket connection

        Returns:
            Set of subscription keys
        """
        return self.websocket_subscriptions.get(websocket, set())


# Global connection manager instance
manager = ConnectionManager()

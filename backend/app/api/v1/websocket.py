"""
WebSocket endpoints for real-time updates.
"""
import json
from typing import Annotated
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.websocket import manager
from app.core.database import get_db_session
from app.core.auth import decode_token
from app.models.database import User, TrainingJob, MLModel, BatchPredictionJob
from app.core.permissions import has_resource_access
from app.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


async def safe_websocket_close(websocket: WebSocket, code: int = status.WS_1000_NORMAL_CLOSURE, reason: str = ""):
    """
    Safely close a WebSocket connection, handling cases where it's already closed.
    
    Args:
        websocket: WebSocket connection to close
        code: Close code
        reason: Close reason
    """
    try:
        # Try to close the websocket
        await websocket.close(code=code, reason=reason)
    except (AttributeError, RuntimeError, ConnectionError) as e:
        # WebSocket is already closed or in invalid state - this is fine
        logger.debug(f"WebSocket already closed or invalid state: {e}")
    except Exception as e:
        # Log unexpected errors but don't raise
        logger.debug(f"Error closing WebSocket (likely already closed): {e}")


async def get_user_from_websocket_token(
    token: str,
    db: AsyncSession
) -> User | None:
    """
    Authenticate user from WebSocket token.

    Args:
        token: JWT token
        db: Database session

    Returns:
        User if authenticated, None otherwise
    """
    # Decode token
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    # Get user ID from token
    user_id_str = payload.get("sub")
    if not user_id_str:
        return None

    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        return None

    # Get user from database
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        return None

    # Load all attributes before returning
    _ = (user.id, user.email, user.username, user.hashed_password,
         user.full_name, user.role, user.is_active, user.is_superuser,
         user.created_at, user.updated_at)

    return user


async def check_job_access(
    job_id: int,
    user: User,
    db: AsyncSession
) -> bool:
    """
    Check if user has access to a training job.

    Args:
        job_id: Training job ID
        user: User requesting access
        db: Database session

    Returns:
        True if user has access, False otherwise
    """
    # Get training job
    result = await db.execute(
        select(TrainingJob).where(TrainingJob.id == job_id)
    )
    job = result.scalar_one_or_none()

    if not job:
        return False

    # Check if user has access (admin, superuser, or owner)
    if user.is_superuser or user.role == "admin":
        return True

    if job.created_by == user.id:
        return True

    return False


@router.websocket("/ws/training/{job_id}")
async def websocket_training_endpoint(
    websocket: WebSocket,
    job_id: int,
    token: Annotated[str, Query()]
):
    """
    WebSocket endpoint for real-time training job updates.

    Args:
        websocket: WebSocket connection
        job_id: Training job ID to subscribe to
        token: JWT authentication token (query parameter)
    """
    # Accept the websocket connection first
    await websocket.accept()
    
    subscription_key = None
    user = None
    try:
        # Authenticate user with a short-lived database session
        async with get_db_session() as db:
            user = await get_user_from_websocket_token(token, db)
            if not user:
                await safe_websocket_close(websocket, code=status.WS_1008_POLICY_VIOLATION)
                logger.warning(f"WebSocket connection rejected: invalid token for job {job_id}")
                return

            # Check if user has access to this job
            has_access = await check_job_access(job_id, user, db)
            if not has_access:
                await safe_websocket_close(websocket, code=status.WS_1008_POLICY_VIOLATION)
                logger.warning(
                    f"WebSocket connection rejected: user {user.id} has no access to job {job_id}"
                )
                return
        
        # Session is now closed - we don't hold it for the entire WebSocket connection
        # Connect with subscription key (don't accept again, already accepted)
        if user:  # Ensure user is set before connecting
            subscription_key = f"training_{job_id}"
            await manager.connect(websocket, subscription_key, user.id, accept=False)

        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connected",
                "message": "Connected to training job updates",
                "job_id": job_id
            }),
            websocket
        )

        # Keep connection alive and handle incoming messages
        while True:
            # Wait for any message from client (ping/pong or subscription commands)
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                # Handle subscription commands
                if message.get("type") == "subscribe":
                    new_subscription = message.get("subscription_key")
                    if new_subscription:
                        await manager.subscribe(websocket, new_subscription, user.id)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "subscribed",
                                "subscription_key": new_subscription
                            }),
                            websocket
                        )
                elif message.get("type") == "unsubscribe":
                    subscription_to_remove = message.get("subscription_key")
                    if subscription_to_remove:
                        await manager.unsubscribe(websocket, subscription_to_remove)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "unsubscribed",
                                "subscription_key": subscription_to_remove
                            }),
                            websocket
                        )
            except json.JSONDecodeError:
                # Handle ping/pong for keepalive (plain text)
                if data == "ping":
                    await manager.send_personal_message("pong", websocket)

    except WebSocketDisconnect:
        if subscription_key:
            await manager.disconnect(websocket, subscription_key)
        logger.info(f"WebSocket disconnected for job {job_id}")

    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {e}", exc_info=True)
        try:
            if subscription_key:
                await manager.disconnect(websocket, subscription_key)
        except:
            pass


@router.websocket("/ws/predictions/{model_id}")
async def websocket_predictions_endpoint(
    websocket: WebSocket,
    model_id: int,
    token: Annotated[str, Query()]
):
    """
    WebSocket endpoint for real-time prediction streaming.

    Args:
        websocket: WebSocket connection
        model_id: Model ID to subscribe to predictions for
        token: JWT authentication token (query parameter)
    """
    # Accept the websocket connection first
    await websocket.accept()
    
    subscription_key = None
    user = None
    try:
        # Authenticate user with a short-lived database session
        async with get_db_session() as db:
            user = await get_user_from_websocket_token(token, db)
            if not user:
                await safe_websocket_close(websocket, code=status.WS_1008_POLICY_VIOLATION)
                logger.warning(f"WebSocket connection rejected: invalid token for model {model_id}")
                return

            # Check if user has access to this model
            model = await db.get(MLModel, model_id)
            if not model:
                await safe_websocket_close(websocket, code=status.WS_1008_POLICY_VIOLATION)
                logger.warning(f"WebSocket connection rejected: model {model_id} not found")
                return
            
            if not has_resource_access(user, model.created_by, model.shared_with):
                await safe_websocket_close(websocket, code=status.WS_1008_POLICY_VIOLATION)
                logger.warning(
                    f"WebSocket connection rejected: user {user.id} has no access to model {model_id}"
                )
                return
        
        # Session is now closed - we don't hold it for the entire WebSocket connection
        # Connect with subscription key (don't accept again, already accepted)
        if user:  # Ensure user is set before connecting
            subscription_key = f"model_{model_id}"
            await manager.connect(websocket, subscription_key, user.id, accept=False)

        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connected",
                "message": "Connected to prediction stream",
                "model_id": model_id
            }),
            websocket
        )

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                # Handle subscription commands
                if message.get("type") == "subscribe":
                    new_subscription = message.get("subscription_key")
                    if new_subscription:
                        await manager.subscribe(websocket, new_subscription, user.id)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "subscribed",
                                "subscription_key": new_subscription
                            }),
                            websocket
                        )
                elif message.get("type") == "unsubscribe":
                    subscription_to_remove = message.get("subscription_key")
                    if subscription_to_remove:
                        await manager.unsubscribe(websocket, subscription_to_remove)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "unsubscribed",
                                "subscription_key": subscription_to_remove
                            }),
                            websocket
                        )
            except json.JSONDecodeError:
                # Handle ping/pong for keepalive (plain text)
                if data == "ping":
                    await manager.send_personal_message("pong", websocket)

    except WebSocketDisconnect:
        if subscription_key:
            await manager.disconnect(websocket, subscription_key)
        logger.info(f"WebSocket disconnected for model {model_id}")

    except Exception as e:
        logger.error(f"WebSocket error for model {model_id}: {e}", exc_info=True)
        try:
            if subscription_key:
                await manager.disconnect(websocket, subscription_key)
        except:
            pass


@router.websocket("/ws/dashboard")
async def websocket_dashboard_endpoint(
    websocket: WebSocket,
    token: Annotated[str, Query()]
):
    """
    WebSocket endpoint for real-time dashboard updates.

    Args:
        websocket: WebSocket connection
        token: JWT authentication token (query parameter)
    """
    # Accept the websocket connection first
    await websocket.accept()
    
    subscription_key = None
    user = None
    try:
        # Authenticate user with a short-lived database session
        async with get_db_session() as db:
            user = await get_user_from_websocket_token(token, db)
            if not user:
                await safe_websocket_close(websocket, code=status.WS_1008_POLICY_VIOLATION)
                logger.warning("WebSocket connection rejected: invalid token for dashboard")
                return
        
        # Session is now closed - we don't hold it for the entire WebSocket connection
        # Connect with subscription key (don't accept again, already accepted)
        if user:  # Ensure user is set before connecting
            subscription_key = f"dashboard_user_{user.id}"
            await manager.connect(websocket, subscription_key, user.id, accept=False)

        # Send initial connection confirmation
        await manager.send_personal_message(
            json.dumps({
                "type": "connected",
                "message": "Connected to dashboard updates",
                "user_id": user.id
            }),
            websocket
        )

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                # Handle subscription commands (e.g., subscribe to specific models, batch jobs, etc.)
                if message.get("type") == "subscribe":
                    new_subscription = message.get("subscription_key")
                    if new_subscription:
                        await manager.subscribe(websocket, new_subscription, user.id)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "subscribed",
                                "subscription_key": new_subscription
                            }),
                            websocket
                        )
                elif message.get("type") == "unsubscribe":
                    subscription_to_remove = message.get("subscription_key")
                    if subscription_to_remove:
                        await manager.unsubscribe(websocket, subscription_to_remove)
                        await manager.send_personal_message(
                            json.dumps({
                                "type": "unsubscribed",
                                "subscription_key": subscription_to_remove
                            }),
                            websocket
                        )
            except json.JSONDecodeError:
                # Handle ping/pong for keepalive (plain text)
                if data == "ping":
                    await manager.send_personal_message("pong", websocket)

    except WebSocketDisconnect:
        if subscription_key:
            await manager.disconnect(websocket, subscription_key)
        logger.info(f"WebSocket disconnected for dashboard")

    except Exception as e:
        logger.error(f"WebSocket error for dashboard: {e}", exc_info=True)
        try:
            if subscription_key:
                await manager.disconnect(websocket, subscription_key)
        except:
            pass

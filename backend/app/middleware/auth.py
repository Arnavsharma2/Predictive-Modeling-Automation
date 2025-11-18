"""
Authentication middleware and dependencies for FastAPI.
"""
from datetime import datetime, timezone
from typing import Annotated, Optional
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.core.auth import decode_token, verify_api_key
from app.core.config import settings
from app.models.database import User, APIKey

# Security schemes
security = HTTPBearer(auto_error=False)


async def get_current_user_from_token(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> Optional[User]:
    """
    Get current user from JWT token.

    Args:
        credentials: HTTP bearer credentials
        db: Database session

    Returns:
        User if authenticated, None otherwise
    """
    if not credentials:
        return None

    token = credentials.credentials

    # Decode token
    payload = decode_token(token)
    if not payload or payload.get("type") != "access":
        return None

    # Get user ID from token (convert from string to int)
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

    if not user:
        return None

    # Access ALL attributes to load them while session is active
    # This must be done before expunging to ensure they're available later
    _ = (user.id, user.email, user.username, user.hashed_password,
         user.full_name, user.role, user.is_active, user.is_superuser,
         user.created_at, user.updated_at)

    if not user.is_active:
        return None

    # Detach the user from the session so it can be used after session closes
    db.expunge(user)

    return user


async def get_current_user_from_api_key(
    db: Annotated[AsyncSession, Depends(get_db)],
    api_key: Optional[str] = Header(None, alias=settings.API_KEY_HEADER)
) -> Optional[User]:
    """
    Get current user from API key.

    Args:
        api_key: API key from header
        db: Database session

    Returns:
        User if authenticated, None otherwise
    """
    if not api_key:
        return None

    # Extract key prefix (first 12 characters)
    if len(api_key) < 12:
        return None

    key_prefix = api_key[:12]

    # Find API key by prefix
    result = await db.execute(
        select(APIKey).where(
            APIKey.key_prefix == key_prefix,
            APIKey.is_active == True
        )
    )
    db_api_key = result.scalar_one_or_none()

    if not db_api_key:
        return None

    # Verify API key
    if not verify_api_key(api_key, db_api_key.hashed_key):
        return None

    # Check expiration
    if db_api_key.expires_at and db_api_key.expires_at < datetime.now(timezone.utc):
        return None

    # Update last used timestamp
    db_api_key.last_used_at = datetime.now(timezone.utc)
    await db.flush()  # Flush changes without committing

    # Get user
    result = await db.execute(
        select(User).where(User.id == db_api_key.user_id)
    )
    user = result.scalar_one_or_none()

    if not user:
        return None

    # Access ALL attributes to load them while session is active
    # This must be done before expunging to ensure they're available later
    _ = (user.id, user.email, user.username, user.hashed_password,
         user.full_name, user.role, user.is_active, user.is_superuser,
         user.created_at, user.updated_at)

    if not user.is_active:
        return None

    # Detach the user from the session so it can be used after session closes
    # All attributes are now loaded in memory
    db.expunge(user)

    return user


async def get_current_user(
    user_from_token: Annotated[Optional[User], Depends(get_current_user_from_token)],
    user_from_api_key: Annotated[Optional[User], Depends(get_current_user_from_api_key)]
) -> User:
    """
    Get current authenticated user from either JWT token or API key.

    Args:
        user_from_token: User from JWT token
        user_from_api_key: User from API key

    Returns:
        Authenticated user

    Raises:
        HTTPException: If not authenticated
    """
    # Try token first, then API key
    user = user_from_token or user_from_api_key

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Get current active user.

    Args:
        current_user: Current user

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_superuser(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Get current superuser.

    Args:
        current_user: Current user

    Returns:
        Superuser

    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


# Optional authentication (doesn't raise error if not authenticated)
async def get_current_user_optional(
    user_from_token: Annotated[Optional[User], Depends(get_current_user_from_token)],
    user_from_api_key: Annotated[Optional[User], Depends(get_current_user_from_api_key)]
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise return None.

    Args:
        user_from_token: User from JWT token
        user_from_api_key: User from API key

    Returns:
        User if authenticated, None otherwise
    """
    return user_from_token or user_from_api_key

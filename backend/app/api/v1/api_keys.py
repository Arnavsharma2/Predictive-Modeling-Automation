"""
API key management endpoints.
"""
from datetime import datetime
from typing import Annotated, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.auth import create_api_key, hash_api_key
from app.middleware.auth import get_current_user
from app.models.database import User, APIKey
from app.models.schemas.auth import APIKeyCreate, APIKeyResponse, APIKeyListItem

router = APIRouter(prefix="/api-keys", tags=["API Keys"])


@router.post("", response_model=APIKeyResponse, status_code=status.HTTP_201_CREATED)
async def create_user_api_key(
    api_key_data: APIKeyCreate,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Create a new API key for the authenticated user.

    Args:
        api_key_data: API key creation data
        current_user: Authenticated user
        db: Database session

    Returns:
        Created API key (plain key only returned once)
    """
    # Generate API key
    plain_key = create_api_key()
    key_prefix = plain_key[:12]  # e.g., "apk_abc123"
    hashed_key = hash_api_key(plain_key)

    # Create API key record
    db_api_key = APIKey(
        name=api_key_data.name,
        key_prefix=key_prefix,
        hashed_key=hashed_key,
        user_id=current_user.id,
        expires_at=api_key_data.expires_at,
        is_active=True
    )

    db.add(db_api_key)
    await db.commit()
    await db.refresh(db_api_key)

    # Return response with plain key (only time it's visible)
    return APIKeyResponse(
        id=db_api_key.id,
        name=db_api_key.name,
        api_key=plain_key,
        key_prefix=key_prefix,
        expires_at=db_api_key.expires_at,
        created_at=db_api_key.created_at
    )


@router.get("", response_model=List[APIKeyListItem])
async def list_api_keys(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    List all API keys for the authenticated user.

    Args:
        current_user: Authenticated user
        db: Database session

    Returns:
        List of API keys (without the actual key values)
    """
    result = await db.execute(
        select(APIKey).where(APIKey.user_id == current_user.id).order_by(APIKey.created_at.desc())
    )
    api_keys = result.scalars().all()

    return api_keys


@router.delete("/{api_key_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_api_key(
    api_key_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Delete an API key.

    Args:
        api_key_id: ID of the API key to delete
        current_user: Authenticated user
        db: Database session

    Raises:
        HTTPException: If API key not found or doesn't belong to user
    """
    # Find API key
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == api_key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Delete API key
    await db.delete(api_key)
    await db.commit()


@router.patch("/{api_key_id}/deactivate", response_model=APIKeyListItem)
async def deactivate_api_key(
    api_key_id: int,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Deactivate an API key without deleting it.

    Args:
        api_key_id: ID of the API key to deactivate
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated API key

    Raises:
        HTTPException: If API key not found or doesn't belong to user
    """
    # Find API key
    result = await db.execute(
        select(APIKey).where(
            APIKey.id == api_key_id,
            APIKey.user_id == current_user.id
        )
    )
    api_key = result.scalar_one_or_none()

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )

    # Deactivate API key
    api_key.is_active = False
    await db.commit()
    await db.refresh(api_key)

    return api_key

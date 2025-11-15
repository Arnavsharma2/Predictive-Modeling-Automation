"""
Authentication endpoints for login, register, and token refresh.
"""
from datetime import datetime, timedelta
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import get_db
from app.core.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    decode_token
)
from app.middleware.auth import get_current_user
from app.models.database import User
from app.models.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserResponse,
    RefreshTokenRequest,
    ChangePasswordRequest
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Register a new user.

    Args:
        user_data: User registration data
        db: Database session

    Returns:
        Created user data with access and refresh tokens

    Raises:
        HTTPException: If username or email already exists
    """
    # Check if username exists
    result = await db.execute(
        select(User).where(User.username == user_data.username)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Check if email exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create new user
    hashed_password = get_password_hash(user_data.password)
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
        is_active=True
    )

    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)

    # Create tokens for the new user
    token_data = {
        "sub": str(db_user.id),
        "username": db_user.username,
        "role": db_user.role.value
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(db_user.id)})

    # Return user data with tokens (matching frontend expectations)
    return {
        "user": UserResponse.model_validate(db_user),
        "access_token": access_token,
        "refresh_token": refresh_token
    }


@router.post("/login", response_model=TokenResponse)
async def login(
    login_data: LoginRequest,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Login and receive access and refresh tokens.

    Args:
        login_data: Login credentials (username/email and password)
        db: Database session

    Returns:
        Access and refresh tokens

    Raises:
        HTTPException: If credentials are invalid
    """
    # Find user by username or email
    result = await db.execute(
        select(User).where(
            (User.username == login_data.username) | (User.email == login_data.username)
        )
    )
    user = result.scalar_one_or_none()

    # Verify user exists and password is correct
    if not user or not verify_password(login_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated"
        )

    # Create tokens
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value
    }

    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(user.id)})

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800  # 30 minutes
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_data: RefreshTokenRequest,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Refresh access token using a refresh token.

    Args:
        refresh_data: Refresh token
        db: Database session

    Returns:
        New access and refresh tokens

    Raises:
        HTTPException: If refresh token is invalid
    """
    # Decode refresh token
    payload = decode_token(refresh_data.refresh_token)

    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user (convert user ID from string to int)
    user_id_str = payload.get("sub")
    try:
        user_id = int(user_id_str)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )

    # Create new tokens
    token_data = {
        "sub": str(user.id),
        "username": user.username,
        "role": user.role.value
    }

    access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token({"sub": str(user.id)})

    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        expires_in=1800
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)]
):
    """
    Get current authenticated user's information.

    Args:
        current_user: Authenticated user from token

    Returns:
        Current user data
    """
    return current_user


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Change user password.

    Args:
        password_data: Current and new password
        current_user: Authenticated user
        db: Database session

    Returns:
        Success message

    Raises:
        HTTPException: If current password is incorrect
    """
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )

    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    await db.commit()

    return {"message": "Password changed successfully"}

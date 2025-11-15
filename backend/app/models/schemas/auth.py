"""
Authentication schemas for request/response validation.
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, validator
from app.models.database.users import UserRole


# User schemas
class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = None


class UserCreate(UserBase):
    """User creation schema."""
    password: str = Field(..., min_length=8, max_length=100)
    role: UserRole = UserRole.VIEWER


class UserUpdate(BaseModel):
    """User update schema."""
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = None
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserResponse(UserBase):
    """User response schema."""
    id: int
    role: UserRole
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# Authentication schemas
class LoginRequest(BaseModel):
    """Login request schema."""
    username: str  # Can be username or email
    password: str


class TokenResponse(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes in seconds


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


class TokenPayload(BaseModel):
    """Token payload schema."""
    sub: int  # User ID
    username: str
    role: UserRole
    exp: datetime
    iat: datetime
    type: str  # "access" or "refresh"


# API Key schemas
class APIKeyCreate(BaseModel):
    """API key creation schema."""
    name: str = Field(..., min_length=1, max_length=255)
    expires_at: Optional[datetime] = None


class APIKeyResponse(BaseModel):
    """API key response schema (returned only on creation)."""
    id: int
    name: str
    api_key: str  # Plain key, only returned once
    key_prefix: str
    expires_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class APIKeyListItem(BaseModel):
    """API key list item schema (without the actual key)."""
    id: int
    name: str
    key_prefix: str
    is_active: bool
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ChangePasswordRequest(BaseModel):
    """Change password request schema."""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)

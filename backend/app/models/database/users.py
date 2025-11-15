"""
User database model.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base
import enum


class UserRole(str, enum.Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    DATA_SCIENTIST = "data_scientist"
    VIEWER = "viewer"


class User(Base):
    """User model."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    role = Column(Enum(UserRole, values_callable=lambda obj: [e.value for e in obj]), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")


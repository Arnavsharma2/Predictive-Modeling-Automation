"""
API key database model for programmatic access.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class APIKey(Base):
    """API Key model for programmatic access."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)  # User-friendly name for the key
    key_prefix = Column(String(10), index=True, nullable=False)  # First 8 chars for identification
    hashed_key = Column(String(255), nullable=False)  # Hashed full key
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    is_active = Column(Boolean, default=True)
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)  # Optional expiration
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="api_keys")

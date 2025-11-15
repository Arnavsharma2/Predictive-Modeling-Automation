"""
Data source database model.
"""
from sqlalchemy import Column, Integer, String, JSON, DateTime, Enum as SQLEnum, ForeignKey
from sqlalchemy.sql import func
import enum
from app.core.database import Base


class DataSourceType(str, enum.Enum):
    """Data source type enumeration."""
    CSV = "csv"
    API = "api"
    WEB_SCRAPE = "web_scrape"
    CLOUD_STORAGE = "cloud_storage"


class DataSourceStatus(str, enum.Enum):
    """Data source status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PROCESSING = "processing"


class DataSource(Base):
    """Data source model."""
    
    __tablename__ = "data_sources"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False, index=True)
    type = Column(SQLEnum(DataSourceType), nullable=False)
    status = Column(SQLEnum(DataSourceStatus), default=DataSourceStatus.ACTIVE)
    config = Column(JSON, nullable=True)  # Store source-specific configuration
    description = Column(String(1000), nullable=True)
    
    # User ownership
    created_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    shared_with = Column(JSON, nullable=True)  # JSON array of user IDs or team IDs that have access
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


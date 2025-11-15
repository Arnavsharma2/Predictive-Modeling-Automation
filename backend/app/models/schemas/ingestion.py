"""
Pydantic schemas for data ingestion.
"""
from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, validator
from app.models.database.data_sources import DataSourceType, DataSourceStatus
from app.models.database.etl_jobs import ETLJobStatus


# CSV Upload Schemas
class CSVUploadRequest(BaseModel):
    """Schema for CSV file upload request."""
    name: str = Field(..., min_length=1, max_length=255, description="Name for the data source")
    description: Optional[str] = Field(None, max_length=1000, description="Optional description")
    auto_process: bool = Field(True, description="Automatically trigger ETL pipeline after upload")


class CSVUploadResponse(BaseModel):
    """Schema for CSV file upload response."""
    source_id: int
    name: str
    status: DataSourceStatus
    message: str
    file_path: Optional[str] = None


# API Source Schemas
class APIAuthConfig(BaseModel):
    """API authentication configuration."""
    auth_type: str = Field(..., description="Authentication type: 'api_key', 'oauth', 'bearer', 'basic'")
    api_key: Optional[str] = Field(None, description="API key (for api_key auth)")
    api_key_header: Optional[str] = Field("X-API-Key", description="Header name for API key")
    bearer_token: Optional[str] = Field(None, description="Bearer token (for bearer auth)")
    username: Optional[str] = Field(None, description="Username (for basic auth)")
    password: Optional[str] = Field(None, description="Password (for basic auth)")
    oauth_token_url: Optional[HttpUrl] = Field(None, description="OAuth token URL")
    oauth_client_id: Optional[str] = Field(None, description="OAuth client ID")
    oauth_client_secret: Optional[str] = Field(None, description="OAuth client secret")


class APISourceRequest(BaseModel):
    """Schema for API source configuration request."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    url: HttpUrl = Field(..., description="API endpoint URL")
    method: str = Field("GET", description="HTTP method: GET, POST, etc.")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional HTTP headers")
    params: Optional[Dict[str, Any]] = Field(None, description="Query parameters")
    body: Optional[Dict[str, Any]] = Field(None, description="Request body (for POST/PUT)")
    auth: Optional[APIAuthConfig] = Field(None, description="Authentication configuration")
    rate_limit: Optional[int] = Field(None, description="Rate limit (requests per minute)")
    retry_count: int = Field(3, ge=0, le=10, description="Number of retry attempts")
    retry_delay: int = Field(1, ge=1, le=60, description="Delay between retries (seconds)")
    polling_interval: Optional[int] = Field(None, ge=60, description="Polling interval in seconds (for scheduled ingestion)")
    auto_process: bool = Field(True, description="Automatically trigger ETL pipeline after data fetch")


class APISourceResponse(BaseModel):
    """Schema for API source configuration response."""
    source_id: int
    name: str
    status: DataSourceStatus
    message: str


# Web Scraping Schemas
class ScrapingConfig(BaseModel):
    """Web scraping configuration."""
    url: HttpUrl = Field(..., description="URL to scrape")
    selectors: Dict[str, str] = Field(..., description="CSS selectors for data extraction")
    pagination: Optional[Dict[str, Any]] = Field(None, description="Pagination configuration")
    respect_robots_txt: bool = Field(True, description="Respect robots.txt")
    rate_limit: Optional[int] = Field(1, ge=1, le=60, description="Delay between requests (seconds)")
    max_pages: Optional[int] = Field(None, ge=1, description="Maximum pages to scrape")


class ScrapingRequest(BaseModel):
    """Schema for web scraping job request."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=1000)
    config: ScrapingConfig
    auto_process: bool = Field(True, description="Automatically trigger ETL pipeline after scraping")


class ScrapingResponse(BaseModel):
    """Schema for web scraping job response."""
    source_id: int
    name: str
    status: DataSourceStatus
    message: str


# Data Source Response Schemas
class DataSourceResponse(BaseModel):
    """Schema for data source response."""
    id: int
    name: str
    type: DataSourceType
    status: DataSourceStatus
    description: Optional[str]
    config: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class DataSourceListResponse(BaseModel):
    """Schema for list of data sources."""
    sources: List[DataSourceResponse]
    total: int


# ETL Job Schemas
class ETLJobResponse(BaseModel):
    """Schema for ETL job response."""
    id: int
    source_id: int
    status: ETLJobStatus
    prefect_flow_run_id: Optional[str]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    error_message: Optional[str]
    records_processed: int
    records_failed: int
    metadata: Optional[Dict[str, Any]] = Field(None, alias="meta_data")
    created_at: datetime
    updated_at: Optional[datetime]
    
    model_config = {"protected_namespaces": (), "from_attributes": True, "populate_by_name": True}


class ETLJobListResponse(BaseModel):
    """Schema for list of ETL jobs."""
    jobs: List[ETLJobResponse]
    total: int


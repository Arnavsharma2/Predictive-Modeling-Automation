"""
Application configuration management using Pydantic settings.
"""
import json
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Application
    APP_NAME: str = "AI-Powered Analytics Platform"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "local"  # local, staging, production
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:3001"]
    
    @field_validator('CORS_ORIGINS', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS_ORIGINS from environment variable.
        
        Supports:
        - JSON array: '["https://example.com","https://app.example.com"]'
        - Comma-separated: 'https://example.com,https://app.example.com'
        - Single string: 'https://example.com'
        """
        if isinstance(v, str):
            # Try parsing as JSON first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            
            # Try comma-separated values
            if ',' in v:
                return [origin.strip() for origin in v.split(',') if origin.strip()]
            
            # Single value
            return [v.strip()] if v.strip() else []
        
        return v

    # Security
    SECRET_KEY: str = "your-secret-key-here-change-in-production"  # Change in production!
    API_KEY_HEADER: str = "X-API-Key"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@db:5432/analytics_db"
    DB_POOL_SIZE: int = 20  # Increased for WebSocket connections and concurrent requests
    DB_MAX_OVERFLOW: int = 30  # Increased to handle bursts
    
    # Redis
    REDIS_URL: str = "redis://redis:6379/0"
    REDIS_CACHE_TTL: int = 3600
    
    # Cloud Storage - AWS S3
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: Optional[str] = None

    # Cloud Storage - Azure Blob
    AZURE_STORAGE_ACCOUNT_NAME: Optional[str] = None
    AZURE_STORAGE_ACCOUNT_KEY: Optional[str] = None
    AZURE_STORAGE_CONTAINER_NAME: Optional[str] = None

    # Cloud Storage - Local
    LOCAL_STORAGE_PATH: str = "/app/storage"

    # Cloud Storage Provider (s3, azure, or local)
    CLOUD_STORAGE_PROVIDER: str = "local"
    
    # Prefect
    PREFECT_API_URL: str = "http://prefect:4200/api"
    
    # MLflow
    MLFLOW_TRACKING_URI: str = "http://mlflow:5000"  # Internal Docker network port (5000), external is 5001
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"  # json or text

    # Monitoring and Observability
    SENTRY_DSN: Optional[str] = None  # Set in production for error tracking
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    ENABLE_TRACE_CONSOLE_EXPORT: bool = False  # Set to True to see traces in console (very verbose)


settings = Settings()


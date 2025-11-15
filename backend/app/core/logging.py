"""
Logging configuration.
"""
import logging
import sys
from typing import Any
import json
from datetime import datetime

from app.core.config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)
        
        return json.dumps(log_data)


def setup_logging() -> None:
    """Configure application logging."""
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    # Create formatter
    if settings.LOG_FORMAT == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers to prevent duplicates
    root_logger.handlers.clear()
    
    # Also clear handlers from child loggers that might have been configured
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "sqlalchemy.engine"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True  # Allow propagation to root logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    root_logger.addHandler(console_handler)
    
    # Set levels for third-party libraries
    # SQLAlchemy: Only log warnings and errors, not INFO queries
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
    
    # Uvicorn: Set appropriate levels
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    
    # AWS/Boto: Reduce verbosity
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


"""
Database connection and session management.
"""
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import declarative_base
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from sqlalchemy.exc import OperationalError, DisconnectionError

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Create async engine with connection pooling for FastAPI
# Default pool for asyncpg is AsyncAdaptedQueuePool which efficiently manages connections
# This prevents connection exhaustion by reusing connections from a pool
engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,  # Number of connections to maintain
    max_overflow=settings.DB_MAX_OVERFLOW,  # Additional connections beyond pool_size
    pool_pre_ping=False,  # Disabled to avoid event loop conflicts with Prefect tasks
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False,  # Disable echo - use logging system instead to avoid duplicate logs
    # Connection timeout settings
    connect_args={
        "command_timeout": 30,  # 30 second timeout for queries
        "server_settings": {
            "application_name": "analytics_backend",
            "tcp_keepalives_idle": "600",  # Keep connections alive
            "tcp_keepalives_interval": "30",
            "tcp_keepalives_count": "3",
        },
    },
)

# Create separate engine for Prefect tasks with NullPool
# NullPool creates a new connection for each operation in the current event loop
# This avoids event loop conflicts when Prefect tasks run in different loops
prefect_engine = create_async_engine(
    settings.DATABASE_URL,
    poolclass=NullPool,  # No pooling - each operation gets a fresh connection
    echo=False,
    connect_args={
        "command_timeout": 30,
        "server_settings": {
            "application_name": "analytics_backend_prefect",
            "tcp_keepalives_idle": "600",
            "tcp_keepalives_interval": "30",
            "tcp_keepalives_count": "3",
        },
    },
)

# Create async session factory for FastAPI
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Create async session factory for Prefect tasks
PrefectSessionLocal = async_sessionmaker(
    prefect_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

# Base class for database models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session with retry logic.
    
    Handles transient connection errors with automatic retries.
    """
    max_retries = 3
    retry_delay = 1.0
    
    for attempt in range(max_retries):
        session = None
        try:
            session = AsyncSessionLocal()
            try:
                yield session
                # Only commit if no exception occurred
                await session.commit()
            except (OperationalError, DisconnectionError) as e:
                # Rollback on connection errors
                try:
                    await session.rollback()
                except Exception:
                    pass  # Ignore rollback errors if session is already invalid
                
                # Check if this is a transient error worth retrying
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in [
                    "connection", "timeout", "network", "closed", "lost",
                    "server closed", "connection reset"
                ]):
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database connection error (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                        # Close session before retrying
                        try:
                            await session.close()
                        except Exception:
                            pass
                        continue
                # Not a transient error or out of retries
                logger.error(f"Database error: {e}")
                raise
            except Exception as e:
                # Rollback on any other exception
                try:
                    await session.rollback()
                except Exception:
                    pass  # Ignore rollback errors if session is already invalid
                raise
            finally:
                # Always close the session
                if session:
                    try:
                        await session.close()
                    except Exception:
                        pass  # Ignore close errors
            # Success - break out of retry loop
            break
        except (OperationalError, DisconnectionError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database session creation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            else:
                logger.error(f"Failed to create database session after {max_retries} attempts: {e}")
                raise


def _is_prefect_context():
    """
    Check if we're running in a Prefect task context.
    """
    try:
        from prefect import runtime
        # Check if we're in a Prefect flow or task run
        if hasattr(runtime, 'flow_run') and runtime.flow_run is not None:
            return True
        if hasattr(runtime, 'task_run') and runtime.task_run is not None:
            return True
        return False
    except (ImportError, AttributeError):
        # If Prefect is not available or runtime is not accessible, assume not in Prefect context
        return False


@asynccontextmanager
async def get_db_session():
    """
    Context manager for getting a database session with retry logic.
    Useful for Prefect tasks and Celery workers to ensure proper event loop handling.
    
    This ensures that database operations run in the current event loop,
    which is important when tasks run in different event loops.
    
    For Prefect tasks, uses NullPool to avoid event loop conflicts.
    For FastAPI, uses QueuePool for efficient connection reuse.
    """
    max_retries = 3
    retry_delay = 1.0
    
    # Detect if we're in a Prefect context
    use_prefect_engine = _is_prefect_context()
    session_factory = PrefectSessionLocal if use_prefect_engine else AsyncSessionLocal
    
    # Ensure we're using the current event loop
    # This is important for Prefect tasks which may run in different loops
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running, this shouldn't happen in async context
        # but if it does, we'll create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Create a session - use Prefect engine if in Prefect context, otherwise use main engine
    for attempt in range(max_retries):
        session = None
        try:
            session = session_factory()
            try:
                yield session
                # Only commit if no exception occurred
                await session.commit()
            except (OperationalError, DisconnectionError) as e:
                # Rollback on connection errors
                try:
                    await session.rollback()
                except Exception:
                    pass  # Ignore rollback errors if session is already invalid
                
                # Check if this is a transient error worth retrying
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in [
                    "connection", "timeout", "network", "closed", "lost",
                    "server closed", "connection reset"
                ]):
                    if attempt < max_retries - 1:
                        logger.warning(
                            f"Database connection error in session (attempt {attempt + 1}/{max_retries}): {e}. "
                            f"Retrying in {retry_delay}s..."
                        )
                        await asyncio.sleep(retry_delay * (attempt + 1))
                        # Close session before retrying
                        try:
                            await session.close()
                        except Exception:
                            pass
                        continue
                # Not a transient error or out of retries
                logger.error(f"Database error in session: {e}")
                raise
            except Exception as e:
                # Rollback on any other exception
                try:
                    await session.rollback()
                except Exception:
                    pass  # Ignore rollback errors if session is already invalid
                raise
            finally:
                # Always close the session
                if session:
                    try:
                        await session.close()
                    except Exception:
                        pass  # Ignore close errors
            # Success - break out of retry loop
            break
        except (OperationalError, DisconnectionError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database session creation failed (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay * (attempt + 1))
                continue
            else:
                logger.error(f"Failed to create database session after {max_retries} attempts: {e}")
                raise



# Alias for Celery workers - uses the same context manager
get_async_session_context = get_db_session

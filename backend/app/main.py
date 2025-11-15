"""
FastAPI application entry point with application factory pattern.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.health import router as health_router
from app.core.database import engine, Base
from app.core.sentry import init_sentry
from app.core.tracing import setup_tracing

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Setup Sentry (if configured)
if settings.SENTRY_DSN:
    init_sentry(dsn=settings.SENTRY_DSN)
    logger.info("Sentry error tracking initialized")

# Handle Prefect TerminationSignal exceptions
try:
    from prefect.exceptions import TerminationSignal
except ImportError:
    TerminationSignal = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan events.
    """
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Create database tables (in production, use migrations)
    if settings.ENVIRONMENT == "local":
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    # Clean up interrupted training jobs from previous runs
    from app.services.job_cleanup import JobCleanupService
    from app.core.database import AsyncSessionLocal

    try:
        async with AsyncSessionLocal() as db:
            cleaned_count = await JobCleanupService.cleanup_interrupted_jobs(db)
            if cleaned_count > 0:
                logger.info(f"Startup cleanup: marked {cleaned_count} interrupted job(s) as failed")
    except Exception as e:
        logger.error(f"Failed to cleanup interrupted jobs on startup: {e}")

    # Start Redis pub/sub listener for WebSocket updates from Celery workers
    import asyncio
    from app.core.redis_pubsub import redis_pubsub
    from app.core.websocket import manager as ws_manager

    # Create the task and let it run in the background
    redis_listener_task = asyncio.create_task(redis_pubsub.subscribe_and_forward(ws_manager))

    # Give the task a chance to start
    await asyncio.sleep(0.1)

    logger.info("Started Redis Pub/Sub listener for cross-process WebSocket updates")

    yield

    # Shutdown
    logger.info("Shutting down application")

    # Stop Redis pub/sub listener
    if redis_listener_task:
        redis_listener_task.cancel()
        try:
            await redis_listener_task
        except asyncio.CancelledError:
            pass
    redis_pubsub.close()
    logger.info("Stopped Redis Pub/Sub listener")

    # Optional: Mark all running jobs as interrupted on shutdown
    try:
        async with AsyncSessionLocal() as db:
            from app.models.database.training_jobs import TrainingJob, TrainingJobStatus
            from sqlalchemy import select, update

            result = await db.execute(
                select(TrainingJob).where(
                    TrainingJob.status == TrainingJobStatus.RUNNING
                )
            )
            running_jobs = result.scalars().all()

            if running_jobs:
                logger.info(f"Marking {len(running_jobs)} running job(s) as interrupted for graceful shutdown")
                for job in running_jobs:
                    await db.execute(
                        update(TrainingJob)
                        .where(TrainingJob.id == job.id)
                        .values(
                            error_message="Training interrupted due to application shutdown",
                            updated_at=datetime.now(timezone.utc)
                        )
                    )
                await db.commit()
    except Exception as e:
        logger.error(f"Failed to mark jobs as interrupted on shutdown: {e}")


def create_app() -> FastAPI:
    """
    Application factory function.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="AI-Powered Analytics Platform API",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Setup OpenTelemetry tracing (if enabled)
    if settings.ENABLE_TRACING:
        setup_tracing(app)
        logger.info("OpenTelemetry tracing initialized")

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Error handling middleware
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        # Handle Prefect TerminationSignal (occurs during uvicorn reload)
        if TerminationSignal and isinstance(exc, TerminationSignal):
            logger.warning(
                f"Prefect TerminationSignal caught (likely due to reload): {exc}",
                extra={"path": request.url.path, "method": request.method},
            )
            from fastapi.responses import JSONResponse
            return JSONResponse(
                status_code=503,
                content={"detail": "Service temporarily unavailable due to reload. Please retry."},
            )
        
        logger.error(
            f"Unhandled exception: {exc}",
            extra={"path": request.url.path, "method": request.method},
        )
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": "AI-Powered Analytics Platform API",
            "docs": "/docs",
            "redoc": "/redoc",
            "health": "/health",
            "api_v1": settings.API_V1_PREFIX,
        }
    
    # Include routers
    app.include_router(health_router, tags=["Health"])

    # Metrics endpoint (outside auth)
    from app.api.v1.metrics import router as metrics_router
    app.include_router(metrics_router)

    # Auth routers (outside API_V1_PREFIX for cleaner URLs)
    from app.api.v1.auth import router as auth_router
    from app.api.v1.api_keys import router as api_keys_router
    app.include_router(auth_router, prefix=settings.API_V1_PREFIX)
    app.include_router(api_keys_router, prefix=settings.API_V1_PREFIX)

    # API v1 routers
    from app.api.v1 import (
        ingestion_router, data_router, ml_router,
        alerts_router, ab_test_router
    )
    from app.api.v1.websocket import router as websocket_router

    app.include_router(ingestion_router, prefix=settings.API_V1_PREFIX)
    app.include_router(data_router, prefix=settings.API_V1_PREFIX)
    app.include_router(ml_router, prefix=settings.API_V1_PREFIX)
    app.include_router(alerts_router, prefix=settings.API_V1_PREFIX)
    app.include_router(ab_test_router, prefix=settings.API_V1_PREFIX)
    app.include_router(websocket_router, prefix=settings.API_V1_PREFIX)

    # Middleware
    from app.middleware.rate_limit import RateLimitMiddleware
    from app.middleware.logging import LoggingMiddleware
    from app.middleware.metrics import MetricsMiddleware

    # Add metrics middleware if enabled
    if settings.ENABLE_METRICS:
        app.add_middleware(MetricsMiddleware)

    app.add_middleware(LoggingMiddleware)
    app.add_middleware(
        RateLimitMiddleware,
        default_limit=100,
        default_window=60,
        per_endpoint_limits={
            "/api/v1/ml/predict": (50, 60),  # 50 requests per minute
            "/api/v1/ml/train": (50, 60),  # 50 requests per minute (increased for training operations)
            "/api/v1/ml/train/status": (300, 60),  # 300 requests per minute for status polling (5 req/sec)
            "/api/v1/ml/models": (300, 60),  # 300 requests per minute for models listing (frontend polling)
            "/api/v1/ml/training": (300, 60),  # 300 requests per minute for training job status polling
        }
    )
    
    return app


# Create app instance
app = create_app()


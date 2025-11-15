"""
Celery application for background tasks.
"""
from celery import Celery
from app.core.config import settings

# Create Celery app
celery_app = Celery(
    "analytics_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.training"]
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600 * 4,  # 4 hours max per task
    task_soft_time_limit=3600 * 3.5,  # Soft limit at 3.5 hours
    worker_prefetch_multiplier=1,  # Process one task at a time
    worker_max_tasks_per_child=1,  # Restart worker after each task (prevents memory leaks)
    task_acks_late=True,  # Acknowledge task after completion
    task_reject_on_worker_lost=True,  # Reject task if worker crashes
    result_expires=3600,  # Results expire after 1 hour
)

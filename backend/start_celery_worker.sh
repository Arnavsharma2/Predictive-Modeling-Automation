#!/bin/bash
# Start Celery worker for model training

# Set environment variables
export PYTHONPATH=/app:$PYTHONPATH

# Start Celery worker
celery -A app.core.celery_app worker \
  --loglevel=info \
  --concurrency=1 \
  --max-tasks-per-child=1 \
  --task-events \
  --without-gossip \
  --without-mingle \
  --without-heartbeat \
  -n worker@%h

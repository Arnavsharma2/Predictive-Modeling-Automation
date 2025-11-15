#!/bin/bash
# Run database migrations on Fly.io

echo "Connecting to Fly.io machine to run migrations..."
flyctl ssh console -a automated-predictive-modeling << 'MIGRATE'
cd /app/backend
alembic upgrade head
exit
MIGRATE

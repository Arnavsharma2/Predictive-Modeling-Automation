FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for ML libraries (CatBoost, XGBoost, LightGBM)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    cmake \
    libgomp1 \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Verify critical package structure exists
RUN ls -la app/api/v1/data/ && ls -la app/api/v1/ml/ && \
    test -f app/api/v1/data/__init__.py && test -f app/api/v1/ml/__init__.py

# Expose port
EXPOSE 8000

# Run the application (Render provides PORT env var)
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}


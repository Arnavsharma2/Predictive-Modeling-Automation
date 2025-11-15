"""
Prometheus metrics collection and custom business metrics.
"""
from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_client import CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

# Create a custom registry (can use default if preferred)
registry = CollectorRegistry()

# Application info
app_info = Info('app', 'Application information', registry=registry)
app_info.info({
    'name': 'AI-Powered Analytics Platform',
    'version': '1.0.0'
})

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration in seconds',
    ['method', 'endpoint'],
    registry=registry
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint'],
    registry=registry
)

# ML Model Training Metrics
model_training_total = Counter(
    'model_training_total',
    'Total number of model training jobs',
    ['model_type', 'status'],
    registry=registry
)

model_training_duration_seconds = Histogram(
    'model_training_duration_seconds',
    'Model training duration in seconds',
    ['model_type'],
    buckets=[10, 30, 60, 120, 300, 600, 1800, 3600],  # Up to 1 hour
    registry=registry
)

model_training_in_progress = Gauge(
    'model_training_in_progress',
    'Number of model training jobs in progress',
    ['model_type'],
    registry=registry
)

model_accuracy = Gauge(
    'model_accuracy',
    'Model accuracy score',
    ['model_id', 'model_type', 'metric_name'],
    registry=registry
)

# Prediction Metrics
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made',
    ['model_id', 'model_type'],
    registry=registry
)

prediction_duration_seconds = Histogram(
    'prediction_duration_seconds',
    'Prediction duration in seconds',
    ['model_id', 'model_type'],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
    registry=registry
)

batch_prediction_size = Histogram(
    'batch_prediction_size',
    'Number of samples in batch predictions',
    ['model_id'],
    buckets=[1, 10, 50, 100, 500, 1000, 5000, 10000],
    registry=registry
)

# Data Ingestion Metrics
data_ingestion_total = Counter(
    'data_ingestion_total',
    'Total number of data ingestion jobs',
    ['source_type', 'status'],
    registry=registry
)

data_ingestion_duration_seconds = Histogram(
    'data_ingestion_duration_seconds',
    'Data ingestion duration in seconds',
    ['source_type'],
    registry=registry
)

data_points_ingested_total = Counter(
    'data_points_ingested_total',
    'Total number of data points ingested',
    ['source_id'],
    registry=registry
)

# Database Metrics
db_connections_active = Gauge(
    'db_connections_active',
    'Number of active database connections',
    registry=registry
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration in seconds',
    ['query_type'],
    registry=registry
)

# Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total number of cache hits',
    ['cache_type'],
    registry=registry
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total number of cache misses',
    ['cache_type'],
    registry=registry
)

# Error Metrics
errors_total = Counter(
    'errors_total',
    'Total number of errors',
    ['error_type', 'endpoint'],
    registry=registry
)

# Alert Metrics
alerts_triggered_total = Counter(
    'alerts_triggered_total',
    'Total number of alerts triggered',
    ['alert_type', 'severity'],
    registry=registry
)

# AB Test Metrics
ab_test_predictions_total = Counter(
    'ab_test_predictions_total',
    'Total number of AB test predictions',
    ['test_id', 'variant'],
    registry=registry
)

# Model Drift Metrics
model_drift_detected_total = Counter(
    'model_drift_detected_total',
    'Total number of model drift detections',
    ['model_id', 'drift_type'],
    registry=registry
)

model_drift_score = Gauge(
    'model_drift_score',
    'Model drift score',
    ['model_id', 'drift_metric'],
    registry=registry
)

# API Key Usage Metrics
api_key_requests_total = Counter(
    'api_key_requests_total',
    'Total number of requests using API keys',
    ['key_prefix'],
    registry=registry
)


def get_metrics():
    """
    Get current metrics in Prometheus format.

    Returns:
        Prometheus metrics in text format
    """
    return generate_latest(registry)


def get_metrics_content_type():
    """
    Get the content type for Prometheus metrics.

    Returns:
        Content type string
    """
    return CONTENT_TYPE_LATEST

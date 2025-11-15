"""
OpenTelemetry distributed tracing configuration.
"""
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from typing import Optional

# Optional OTLP exporter - uncomment and install when using a tracing backend
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

from app.core.config import settings

# Global tracer instance
_tracer: Optional[trace.Tracer] = None


def setup_tracing(app):
    """
    Setup OpenTelemetry tracing for the application.

    Args:
        app: FastAPI application instance
    """
    global _tracer

    # Create resource with service information
    resource = Resource.create({
        "service.name": settings.APP_NAME,
        "service.version": settings.APP_VERSION,
        "deployment.environment": settings.ENVIRONMENT,
    })

    # Create tracer provider
    provider = TracerProvider(resource=resource)

    # Add console exporter for development (disabled by default to avoid log flooding)
    # Only enable if explicitly needed for debugging by setting ENABLE_TRACE_CONSOLE_EXPORT=true
    if settings.ENABLE_TRACE_CONSOLE_EXPORT:
        console_exporter = ConsoleSpanExporter()
        console_processor = BatchSpanProcessor(console_exporter)
        provider.add_span_processor(console_processor)

    # Add OTLP exporter for production (if configured)
    # Uncomment and configure when using a tracing backend like Jaeger, Tempo, etc.
    # otlp_exporter = OTLPSpanExporter(
    #     endpoint="http://tempo:4317",
    #     insecure=True
    # )
    # otlp_processor = BatchSpanProcessor(otlp_exporter)
    # provider.add_span_processor(otlp_processor)

    # Set the tracer provider
    trace.set_tracer_provider(provider)

    # Get tracer
    _tracer = trace.get_tracer(__name__)

    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(app)

    return _tracer


def get_tracer() -> trace.Tracer:
    """
    Get the global tracer instance.

    Returns:
        OpenTelemetry tracer
    """
    global _tracer
    if _tracer is None:
        _tracer = trace.get_tracer(__name__)
    return _tracer


def create_span(name: str, attributes: Optional[dict] = None):
    """
    Create a new span for tracing.

    Args:
        name: Span name
        attributes: Optional span attributes

    Returns:
        Span context manager
    """
    tracer = get_tracer()
    span = tracer.start_span(name)

    if attributes:
        for key, value in attributes.items():
            span.set_attribute(key, value)

    return span


def trace_function(name: Optional[str] = None):
    """
    Decorator to trace a function.

    Args:
        name: Optional custom span name

    Returns:
        Decorated function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name):
                return func(*args, **kwargs)

        return wrapper
    return decorator


async def trace_async_function(name: Optional[str] = None):
    """
    Decorator to trace an async function.

    Args:
        name: Optional custom span name

    Returns:
        Decorated function
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            tracer = get_tracer()

            with tracer.start_as_current_span(span_name):
                return await func(*args, **kwargs)

        return wrapper
    return decorator

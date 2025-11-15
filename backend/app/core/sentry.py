"""
Sentry integration for error tracking and performance monitoring.
"""
from typing import Optional

try:
    import sentry_sdk
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration
    from sentry_sdk.integrations.redis import RedisIntegration
    SENTRY_AVAILABLE = True
except ImportError:
    SENTRY_AVAILABLE = False
    # Create dummy functions if sentry_sdk is not available
    sentry_sdk = None
    FastApiIntegration = None
    SqlalchemyIntegration = None
    RedisIntegration = None

from app.core.config import settings


def init_sentry(
    dsn: Optional[str] = None,
    traces_sample_rate: float = 0.1,
    profiles_sample_rate: float = 0.1
):
    """
    Initialize Sentry for error tracking and performance monitoring.

    Args:
        dsn: Sentry DSN (Data Source Name). If None, Sentry will be disabled.
        traces_sample_rate: Percentage of transactions to trace (0.0 to 1.0)
        profiles_sample_rate: Percentage of transactions to profile (0.0 to 1.0)
    """
    # Return early if sentry_sdk is not available
    if not SENTRY_AVAILABLE:
        return
    
    # Only initialize Sentry if DSN is provided
    if not dsn:
        return

    # Adjust sample rates based on environment
    if settings.ENVIRONMENT == "production":
        traces_sample_rate = 0.2  # 20% of transactions in production
        profiles_sample_rate = 0.1  # 10% of transactions in production
    elif settings.ENVIRONMENT == "staging":
        traces_sample_rate = 0.5  # 50% of transactions in staging
        profiles_sample_rate = 0.2  # 20% of transactions in staging
    else:
        # Development/local - capture everything
        traces_sample_rate = 1.0
        profiles_sample_rate = 1.0

    sentry_sdk.init(
        dsn=dsn,
        environment=settings.ENVIRONMENT,
        release=f"{settings.APP_NAME}@{settings.APP_VERSION}",

        # Performance monitoring
        traces_sample_rate=traces_sample_rate,
        profiles_sample_rate=profiles_sample_rate,

        # Integrations
        integrations=[
            FastApiIntegration(
                transaction_style="endpoint",  # Group by endpoint instead of URL
            ),
            SqlalchemyIntegration(),
            RedisIntegration(),
        ],

        # Additional options
        send_default_pii=False,  # Don't send personally identifiable information
        attach_stacktrace=True,  # Attach stack traces to messages
        max_breadcrumbs=50,  # Maximum number of breadcrumbs

        # Custom tags
        _experiments={
            "profiles_sample_rate": profiles_sample_rate,
        },
    )


def capture_exception(error: Exception, context: Optional[dict] = None):
    """
    Manually capture an exception to Sentry.

    Args:
        error: Exception to capture
        context: Optional context data to include
    """
    if not SENTRY_AVAILABLE:
        return
    
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_context(key, value)

        sentry_sdk.capture_exception(error)


def capture_message(message: str, level: str = "info", context: Optional[dict] = None):
    """
    Manually capture a message to Sentry.

    Args:
        message: Message to capture
        level: Message level (debug, info, warning, error, fatal)
        context: Optional context data to include
    """
    if not SENTRY_AVAILABLE:
        return
    
    with sentry_sdk.push_scope() as scope:
        if context:
            for key, value in context.items():
                scope.set_context(key, value)

        sentry_sdk.capture_message(message, level)


def set_user(user_id: int, email: Optional[str] = None, username: Optional[str] = None):
    """
    Set user context for Sentry events.

    Args:
        user_id: User ID
        email: Optional user email
        username: Optional username
    """
    if not SENTRY_AVAILABLE:
        return
    
    sentry_sdk.set_user({
        "id": user_id,
        "email": email,
        "username": username,
    })


def set_tag(key: str, value: str):
    """
    Set a custom tag for Sentry events.

    Args:
        key: Tag key
        value: Tag value
    """
    if not SENTRY_AVAILABLE:
        return
    
    sentry_sdk.set_tag(key, value)


def start_transaction(name: str, op: str = "http.server"):
    """
    Start a new Sentry transaction for performance monitoring.

    Args:
        name: Transaction name
        op: Operation type

    Returns:
        Transaction context manager
    """
    if not SENTRY_AVAILABLE:
        # Return a no-op context manager
        from contextlib import nullcontext
        return nullcontext()
    
    return sentry_sdk.start_transaction(name=name, op=op)

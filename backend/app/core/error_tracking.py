"""
Error tracking and monitoring with Sentry integration.

Provides centralized error tracking, context enrichment, and structured logging.
"""

import logging
from typing import Any, Dict, Optional

import sentry_sdk
import structlog
from sentry_sdk.integrations.asyncio import AsyncioIntegration
from sentry_sdk.integrations.celery import CeleryIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.httpx import HttpxIntegration
from sentry_sdk.integrations.redis import RedisIntegration
from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

from app.core.config import settings

logger = structlog.get_logger(__name__)


def init_sentry() -> None:
    """
    Initialize Sentry SDK for error tracking.

    Integrates with FastAPI, Celery, SQLAlchemy, Redis, and other components.
    Only initializes if SENTRY_DSN is configured.
    """
    if not settings.SENTRY_DSN:
        logger.info("sentry_not_configured", message="SENTRY_DSN not set, skipping initialization")
        return

    try:
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.SENTRY_ENVIRONMENT,
            # Performance monitoring
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
            # Integrations
            integrations=[
                FastApiIntegration(transaction_style="endpoint"),
                SqlalchemyIntegration(),
                RedisIntegration(),
                CeleryIntegration(),
                AsyncioIntegration(),
                HttpxIntegration(),
            ],
            # Add release info if available
            release=f"{settings.APP_NAME}@{settings.APP_VERSION}",
            # Send default PII (can be disabled for privacy)
            send_default_pii=False,
            # Set max breadcrumbs
            max_breadcrumbs=50,
            # Attach stacktrace to messages
            attach_stacktrace=True,
            # Before send callback for filtering
            before_send=before_send_callback,
            # Before breadcrumb callback
            before_breadcrumb=before_breadcrumb_callback,
        )

        logger.info(
            "sentry_initialized",
            environment=settings.SENTRY_ENVIRONMENT,
            release=f"{settings.APP_NAME}@{settings.APP_VERSION}",
        )

    except Exception as e:
        logger.error("sentry_init_failed", error=str(e))


def before_send_callback(event: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter and modify events before sending to Sentry.

    Args:
        event: Sentry event dictionary
        hint: Additional context

    Returns:
        Modified event or None to drop the event
    """
    # Don't send events in test mode
    if settings.DEBUG and "pytest" in event.get("modules", {}):
        return None

    # Filter out specific exceptions
    if "exc_info" in hint:
        exc_type, exc_value, tb = hint["exc_info"]

        # Don't send certain expected errors
        if exc_type.__name__ in ["ValidationError", "NotFoundError"]:
            return None

    # Add custom tags
    event.setdefault("tags", {})
    event["tags"]["app_name"] = settings.APP_NAME
    event["tags"]["app_version"] = settings.APP_VERSION

    return event


def before_breadcrumb_callback(crumb: Dict[str, Any], hint: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Filter and modify breadcrumbs before adding to event.

    Args:
        crumb: Breadcrumb dictionary
        hint: Additional context

    Returns:
        Modified breadcrumb or None to drop it
    """
    # Don't log sensitive query parameters
    if crumb.get("category") == "httplib":
        url = crumb.get("data", {}).get("url", "")
        if "api_key" in url or "password" in url:
            crumb["data"]["url"] = "[FILTERED]"

    return crumb


def capture_exception(
    exception: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "error",
    **kwargs: Any,
) -> str:
    """
    Capture an exception and send to Sentry with additional context.

    Args:
        exception: The exception to capture
        context: Additional context dictionary
        level: Error level (error, warning, info)
        **kwargs: Additional keyword arguments for Sentry

    Returns:
        Event ID from Sentry
    """
    with sentry_sdk.push_scope() as scope:
        # Add context
        if context:
            for key, value in context.items():
                scope.set_context(key, value)

        # Set level
        scope.level = level

        # Add extra kwargs as tags
        for key, value in kwargs.items():
            scope.set_tag(key, str(value))

        # Capture the exception
        event_id = sentry_sdk.capture_exception(exception)

        logger.error(
            "exception_captured",
            exception=str(exception),
            event_id=event_id,
            level=level,
        )

        return event_id


def capture_message(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    level: str = "info",
    **kwargs: Any,
) -> str:
    """
    Capture a message and send to Sentry.

    Args:
        message: The message to capture
        context: Additional context dictionary
        level: Message level (error, warning, info)
        **kwargs: Additional keyword arguments for Sentry

    Returns:
        Event ID from Sentry
    """
    with sentry_sdk.push_scope() as scope:
        # Add context
        if context:
            for key, value in context.items():
                scope.set_context(key, value)

        # Set level
        scope.level = level

        # Add extra kwargs as tags
        for key, value in kwargs.items():
            scope.set_tag(key, str(value))

        # Capture the message
        event_id = sentry_sdk.capture_message(message)

        logger.info(
            "message_captured",
            message=message,
            event_id=event_id,
            level=level,
        )

        return event_id


def set_user_context(user_id: str, email: Optional[str] = None, **kwargs: Any) -> None:
    """
    Set user context for Sentry events.

    Args:
        user_id: User identifier
        email: User email
        **kwargs: Additional user attributes
    """
    sentry_sdk.set_user({"id": user_id, "email": email, **kwargs})


def clear_user_context() -> None:
    """Clear user context."""
    sentry_sdk.set_user(None)


def add_breadcrumb(
    message: str,
    category: str = "custom",
    level: str = "info",
    data: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Add a breadcrumb for debugging context.

    Args:
        message: Breadcrumb message
        category: Breadcrumb category
        level: Breadcrumb level
        data: Additional data dictionary
    """
    sentry_sdk.add_breadcrumb(
        message=message,
        category=category,
        level=level,
        data=data or {},
    )


def set_context(name: str, context: Dict[str, Any]) -> None:
    """
    Set custom context for Sentry events.

    Args:
        name: Context name
        context: Context data dictionary
    """
    sentry_sdk.set_context(name, context)


def set_tag(key: str, value: str) -> None:
    """
    Set a custom tag for Sentry events.

    Args:
        key: Tag key
        value: Tag value
    """
    sentry_sdk.set_tag(key, value)


class SentryContextMiddleware:
    """
    Middleware to add request context to Sentry events.

    Enriches Sentry events with request-specific information.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        """Process request and add context."""
        if scope["type"] == "http":
            # Add request context
            with sentry_sdk.configure_scope() as sentry_scope:
                # Add request info
                sentry_scope.set_context(
                    "request",
                    {
                        "method": scope.get("method"),
                        "path": scope.get("path"),
                        "query_string": scope.get("query_string", b"").decode(),
                    },
                )

                # Add headers (filtered)
                headers = dict(scope.get("headers", []))
                safe_headers = {
                    k.decode(): v.decode()
                    for k, v in headers.items()
                    if k.decode().lower()
                    not in ["authorization", "cookie", "x-api-key"]
                }
                sentry_scope.set_context("headers", safe_headers)

        return await self.app(scope, receive, send)


# Structured logging integration with Sentry
class SentryStructlogProcessor:
    """
    Structlog processor that sends errors to Sentry.

    Integrates structlog with Sentry for structured error tracking.
    """

    def __init__(self, level: int = logging.ERROR):
        self.level = level

    def __call__(self, logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process log event and send to Sentry if needed."""
        # Get log level
        level = event_dict.get("level", "").upper()

        # Send to Sentry if error or critical
        if level in ["ERROR", "CRITICAL", "EXCEPTION"]:
            # Extract exception if present
            exc_info = event_dict.get("exc_info")

            if exc_info:
                # Capture exception
                capture_exception(
                    exc_info[1] if isinstance(exc_info, tuple) else exc_info,
                    context={"event": event_dict},
                    level=level.lower(),
                )
            else:
                # Capture message
                capture_message(
                    event_dict.get("event", ""),
                    context=event_dict,
                    level=level.lower(),
                )

        return event_dict

"""
Rate limiting middleware implementation using SlowAPI and Redis.
"""

import logging
from functools import wraps
from typing import Any, Dict, Optional

import redis.asyncio as redis
from fastapi import HTTPException, Request, status
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from app.core.config import settings

logger = logging.getLogger(__name__)


class RedisRateLimitBackend:
    """Redis backend for rate limiting storage."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis_client = None

    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis for rate limiting: {e}")
            self.redis_client = None

    async def get(self, key: str) -> Optional[int]:
        """Get current count for a key."""
        if not self.redis_client:
            return None

        try:
            value = await self.redis_client.get(key)
            return int(value) if value else 0
        except Exception as e:
            logger.warning(f"Error getting rate limit count: {e}")
            return None

    async def incr(self, key: str, amount: int = 1, expire: int = 3600) -> int:
        """Increment counter and set expiry."""
        if not self.redis_client:
            return 0

        try:
            # Use pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            pipe.incr(key, amount)
            pipe.expire(key, expire)
            results = await pipe.execute()
            return results[0]
        except Exception as e:
            logger.warning(f"Error incrementing rate limit count: {e}")
            return 0

    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


def get_client_identifier(request: Request) -> str:
    """Get client identifier for rate limiting."""
    # Try to get user ID from token if authenticated
    user_id = getattr(request.state, "user_id", None)
    if user_id:
        return f"user:{user_id}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


def get_route_identifier(request: Request) -> str:
    """Get route identifier for rate limiting."""
    return f"{request.method}:{request.url.path}"


# Create rate limiter instance
limiter = Limiter(
    key_func=get_client_identifier,
    default_limits=["1000/hour", "100/minute"],
    storage_uri=settings.REDIS_URL,
    strategy="fixed-window",
)


class CustomRateLimitMiddleware:
    """Custom rate limiting middleware with enhanced features."""

    def __init__(self):
        self.backend = RedisRateLimitBackend(settings.REDIS_URL)
        self.default_limits = {
            "search": {"requests": 100, "window": 3600},  # 100 requests per hour
            "upload": {"requests": 20, "window": 3600},  # 20 uploads per hour
            "default": {"requests": 1000, "window": 3600},  # 1000 requests per hour
        }

    async def __call__(self, request: Request, call_next):
        """Process rate limiting for incoming requests."""
        try:
            # Skip rate limiting for health checks
            if request.url.path in [
                "/health",
                "/api/v1/health",
                "/api/v1/health/detailed",
            ]:
                return await call_next(request)

            # Get client and route identifiers
            client_id = get_client_identifier(request)
            route_id = get_route_identifier(request)

            # Determine rate limit based on route
            rate_limit = self._get_rate_limit_for_route(request.url.path)

            # Check rate limit
            is_allowed = await self._check_rate_limit(
                client_id=client_id,
                route_id=route_id,
                limit=rate_limit["requests"],
                window=rate_limit["window"],
            )

            if not is_allowed:
                logger.warning(f"Rate limit exceeded for {client_id} on {route_id}")
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Rate limit exceeded",
                        "limit": rate_limit["requests"],
                        "window": rate_limit["window"],
                        "retry_after": rate_limit["window"],
                    },
                    headers={"Retry-After": str(rate_limit["window"])},
                )

            # Process request
            response = await call_next(request)

            # Add rate limit headers
            remaining = await self._get_remaining_requests(
                client_id=client_id,
                route_id=route_id,
                limit=rate_limit["requests"],
                window=rate_limit["window"],
            )

            response.headers["X-RateLimit-Limit"] = str(rate_limit["requests"])
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(rate_limit["window"])

            return response

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in rate limiting middleware: {e}")
            # Continue processing if rate limiting fails
            return await call_next(request)

    def _get_rate_limit_for_route(self, path: str) -> Dict[str, int]:
        """Get rate limit configuration for specific route."""
        if "/search" in path:
            return self.default_limits["search"]
        elif "/upload" in path or "/documents" in path:
            return self.default_limits["upload"]
        else:
            return self.default_limits["default"]

    async def _check_rate_limit(
        self, client_id: str, route_id: str, limit: int, window: int
    ) -> bool:
        """Check if request is within rate limit."""
        if not self.backend.redis_client:
            await self.backend.connect()

        # Create unique key for this client and route
        key = f"rate_limit:{client_id}:{route_id}"

        # Increment counter
        current_count = await self.backend.incr(key, 1, window)

        # Check if within limit
        return current_count <= limit

    async def _get_remaining_requests(
        self, client_id: str, route_id: str, limit: int, window: int
    ) -> int:
        """Get remaining requests for client."""
        key = f"rate_limit:{client_id}:{route_id}"
        current_count = await self.backend.get(key) or 0
        return max(0, limit - current_count)


# Custom rate limit exceeded handler
async def custom_rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """Custom handler for rate limit exceeded errors."""
    logger.warning(f"Rate limit exceeded for {get_client_identifier(request)}")

    return HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail={
            "error": "Rate limit exceeded",
            "message": "Too many requests. Please try again later.",
            "retry_after": exc.retry_after,
        },
        headers={"Retry-After": str(exc.retry_after)},
    )


# Decorators for specific endpoints
def rate_limit_search(func):
    """Rate limit decorator for search endpoints."""

    @wraps(func)
    @limiter.limit("100/hour")
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def rate_limit_upload(func):
    """Rate limit decorator for upload endpoints."""

    @wraps(func)
    @limiter.limit("20/hour")
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


def rate_limit_heavy(func):
    """Rate limit decorator for resource-intensive endpoints."""

    @wraps(func)
    @limiter.limit("10/hour")
    async def wrapper(*args, **kwargs):
        return await func(*args, **kwargs)

    return wrapper


# Rate limiting statistics
class RateLimitStats:
    """Rate limiting statistics collector."""

    def __init__(self, backend: RedisRateLimitBackend):
        self.backend = backend

    async def get_client_stats(self, client_id: str) -> Dict[str, Any]:
        """Get rate limiting stats for a specific client."""
        if not self.backend.redis_client:
            return {}

        try:
            # Get all keys for this client
            pattern = f"rate_limit:{client_id}:*"
            keys = []
            async for key in self.backend.redis_client.scan_iter(match=pattern):
                keys.append(key)

            stats = {}
            for key in keys:
                route = key.split(":")[-1]
                count = await self.backend.get(key)
                ttl = await self.backend.redis_client.ttl(key)
                stats[route] = {"current_count": count, "ttl": ttl}

            return stats

        except Exception as e:
            logger.error(f"Error getting rate limit stats: {e}")
            return {}

    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global rate limiting statistics."""
        if not self.backend.redis_client:
            return {}

        try:
            # Count total rate limit keys
            pattern = "rate_limit:*"
            total_keys = 0
            async for _ in self.backend.redis_client.scan_iter(match=pattern):
                total_keys += 1

            return {"total_rate_limited_clients": total_keys, "backend_connected": True}

        except Exception as e:
            logger.error(f"Error getting global rate limit stats: {e}")
            return {"backend_connected": False, "error": str(e)}


# Global instances
rate_limit_middleware = CustomRateLimitMiddleware()
rate_limit_stats = RateLimitStats(rate_limit_middleware.backend)

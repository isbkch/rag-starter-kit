"""
Circuit breaker implementation for external API resilience.

Provides circuit breaker patterns to prevent cascading failures
when external services (OpenAI, vector databases) become unavailable.
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional

import structlog
from pybreaker import CircuitBreaker, CircuitBreakerError

logger = structlog.get_logger(__name__)


class CircuitBreakerConfig:
    """Configuration for circuit breakers."""

    # OpenAI API circuit breaker settings
    OPENAI_FAIL_MAX = 5  # Open circuit after 5 failures
    OPENAI_TIMEOUT = 60  # Try again after 60 seconds
    OPENAI_EXPECTED_EXCEPTION = Exception

    # Vector database circuit breaker settings
    VECTORDB_FAIL_MAX = 3  # More aggressive for vector DBs
    VECTORDB_TIMEOUT = 30  # Shorter timeout for vector DBs
    VECTORDB_EXPECTED_EXCEPTION = Exception

    # Elasticsearch circuit breaker settings
    ELASTICSEARCH_FAIL_MAX = 3
    ELASTICSEARCH_TIMEOUT = 30
    ELASTICSEARCH_EXPECTED_EXCEPTION = Exception


def _on_circuit_open(breaker: CircuitBreaker, *args: Any, **kwargs: Any) -> None:
    """Called when circuit opens (too many failures)."""
    logger.warning(
        "circuit_breaker_opened",
        breaker_name=breaker.name,
        fail_count=breaker.fail_counter,
        message=f"Circuit breaker '{breaker.name}' opened after {breaker.fail_counter} failures",
    )


def _on_circuit_close(breaker: CircuitBreaker, *args: Any, **kwargs: Any) -> None:
    """Called when circuit closes (service recovered)."""
    logger.info(
        "circuit_breaker_closed",
        breaker_name=breaker.name,
        message=f"Circuit breaker '{breaker.name}' closed - service recovered",
    )


def _on_circuit_half_open(breaker: CircuitBreaker, *args: Any, **kwargs: Any) -> None:
    """Called when circuit enters half-open state (testing recovery)."""
    logger.info(
        "circuit_breaker_half_open",
        breaker_name=breaker.name,
        message=f"Circuit breaker '{breaker.name}' half-open - testing service",
    )


# Initialize circuit breakers for external services
openai_breaker = CircuitBreaker(
    fail_max=CircuitBreakerConfig.OPENAI_FAIL_MAX,
    timeout_duration=CircuitBreakerConfig.OPENAI_TIMEOUT,
    expected_exception=CircuitBreakerConfig.OPENAI_EXPECTED_EXCEPTION,
    name="openai_api",
    listeners=[_on_circuit_open, _on_circuit_close, _on_circuit_half_open],
)

vectordb_breaker = CircuitBreaker(
    fail_max=CircuitBreakerConfig.VECTORDB_FAIL_MAX,
    timeout_duration=CircuitBreakerConfig.VECTORDB_TIMEOUT,
    expected_exception=CircuitBreakerConfig.VECTORDB_EXPECTED_EXCEPTION,
    name="vector_database",
    listeners=[_on_circuit_open, _on_circuit_close, _on_circuit_half_open],
)

elasticsearch_breaker = CircuitBreaker(
    fail_max=CircuitBreakerConfig.ELASTICSEARCH_FAIL_MAX,
    timeout_duration=CircuitBreakerConfig.ELASTICSEARCH_TIMEOUT,
    expected_exception=CircuitBreakerConfig.ELASTICSEARCH_EXPECTED_EXCEPTION,
    name="elasticsearch",
    listeners=[_on_circuit_open, _on_circuit_close, _on_circuit_half_open],
)


def async_circuit_breaker(breaker: CircuitBreaker) -> Callable:
    """
    Decorator for async functions to add circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use

    Returns:
        Decorated function with circuit breaker protection

    Example:
        @async_circuit_breaker(openai_breaker)
        async def call_openai_api():
            # API call here
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                # Check circuit state before calling
                if breaker.current_state == "open":
                    logger.warning(
                        "circuit_breaker_blocked_call",
                        breaker_name=breaker.name,
                        function=func.__name__,
                    )
                    raise CircuitBreakerError(
                        f"Circuit breaker '{breaker.name}' is open - service unavailable"
                    )

                # Call the protected function
                result = await func(*args, **kwargs)

                # Record success
                breaker.call_succeeded()
                return result

            except CircuitBreakerError:
                # Re-raise circuit breaker errors
                raise
            except Exception as e:
                # Record failure and re-raise
                breaker.call_failed()
                logger.error(
                    "circuit_breaker_call_failed",
                    breaker_name=breaker.name,
                    function=func.__name__,
                    error=str(e),
                    fail_count=breaker.fail_counter,
                )
                raise

        return wrapper

    return decorator


def sync_circuit_breaker(breaker: CircuitBreaker) -> Callable:
    """
    Decorator for sync functions to add circuit breaker protection.

    Args:
        breaker: CircuitBreaker instance to use

    Returns:
        Decorated function with circuit breaker protection
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return breaker.call(func, *args, **kwargs)

        return wrapper

    return decorator


async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    exceptions: tuple = (Exception,),
) -> Any:
    """
    Retry an async function with exponential backoff.

    Args:
        func: Async function to retry
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        exceptions: Tuple of exceptions to catch and retry

    Returns:
        Result of the function call

    Raises:
        Last exception if all retries fail
    """
    last_exception: Optional[Exception] = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(
                    "retry_exhausted",
                    function=func.__name__,
                    attempts=attempt + 1,
                    error=str(e),
                )
                raise

            # Calculate delay with exponential backoff
            delay = min(initial_delay * (exponential_base**attempt), max_delay)

            logger.warning(
                "retry_attempt",
                function=func.__name__,
                attempt=attempt + 1,
                max_retries=max_retries,
                delay=delay,
                error=str(e),
            )

            await asyncio.sleep(delay)

    # Should never reach here, but for type safety
    if last_exception:
        raise last_exception


def get_circuit_breaker_status() -> dict:
    """
    Get status of all circuit breakers.

    Returns:
        Dictionary with circuit breaker states and statistics
    """
    breakers = {
        "openai": openai_breaker,
        "vectordb": vectordb_breaker,
        "elasticsearch": elasticsearch_breaker,
    }

    status = {}
    for name, breaker in breakers.items():
        status[name] = {
            "state": breaker.current_state,
            "fail_counter": breaker.fail_counter,
            "fail_max": breaker.fail_max,
            "timeout_duration": breaker.timeout_duration,
            "is_available": breaker.current_state != "open",
        }

    return status

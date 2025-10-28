"""
Tests for circuit breaker functionality.
"""

import asyncio

import pytest
from pybreaker import CircuitBreaker, CircuitBreakerError

from app.core.circuit_breaker import (
    async_circuit_breaker,
    get_circuit_breaker_status,
    retry_with_backoff,
)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    def test_circuit_breaker_creation(self):
        """Test that circuit breakers are created with correct configuration."""
        status = get_circuit_breaker_status()

        assert "openai" in status
        assert "vectordb" in status
        assert "elasticsearch" in status

        # Check OpenAI breaker
        openai_status = status["openai"]
        assert openai_status["state"] in ["closed", "open", "half-open"]
        assert openai_status["fail_max"] == 5
        assert openai_status["timeout_duration"] == 60

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_success(self):
        """Test circuit breaker with successful calls."""
        test_breaker = CircuitBreaker(fail_max=3, timeout_duration=5, name="test")
        call_count = 0

        @async_circuit_breaker(test_breaker)
        async def successful_call():
            nonlocal call_count
            call_count += 1
            return "success"

        # Multiple successful calls should work
        for _ in range(5):
            result = await successful_call()
            assert result == "success"

        assert call_count == 5
        assert test_breaker.current_state == "closed"

    @pytest.mark.asyncio
    async def test_async_circuit_breaker_failure(self):
        """Test circuit breaker opens after failures."""
        test_breaker = CircuitBreaker(fail_max=3, timeout_duration=5, name="test_fail")
        call_count = 0

        @async_circuit_breaker(test_breaker)
        async def failing_call():
            nonlocal call_count
            call_count += 1
            raise Exception("Test failure")

        # Make calls that fail
        for i in range(3):
            with pytest.raises(Exception, match="Test failure"):
                await failing_call()

        # Circuit should be open now
        assert test_breaker.current_state == "open"

        # Next call should be blocked by circuit breaker
        with pytest.raises(CircuitBreakerError):
            await failing_call()

        # Call count should not increase when circuit is open
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_success(self):
        """Test retry with backoff for successful call after retries."""
        call_count = 0

        async def eventually_successful():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not ready yet")
            return "success"

        result = await retry_with_backoff(
            eventually_successful,
            max_retries=3,
            initial_delay=0.01,
            exponential_base=2.0,
        )

        assert result == "success"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_retry_with_backoff_failure(self):
        """Test retry with backoff exhausts retries."""
        call_count = 0

        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise ValueError("Always fails")

        with pytest.raises(ValueError, match="Always fails"):
            await retry_with_backoff(
                always_fails,
                max_retries=3,
                initial_delay=0.01,
                exponential_base=2.0,
            )

        # Should have tried: original + 3 retries = 4 times
        assert call_count == 4

    @pytest.mark.asyncio
    async def test_retry_backoff_timing(self):
        """Test that retry backoff increases exponentially."""
        call_times = []

        async def track_timing():
            call_times.append(asyncio.get_event_loop().time())
            raise ValueError("Test")

        try:
            await retry_with_backoff(
                track_timing,
                max_retries=3,
                initial_delay=0.1,
                max_delay=1.0,
                exponential_base=2.0,
            )
        except ValueError:
            pass

        # Should have 4 calls (original + 3 retries)
        assert len(call_times) == 4

        # Check delays are increasing (with some tolerance)
        delays = [call_times[i + 1] - call_times[i] for i in range(len(call_times) - 1)]
        assert delays[0] < delays[1]  # Second delay > first delay
        assert delays[1] < delays[2]  # Third delay > second delay

    def test_get_circuit_breaker_status(self):
        """Test circuit breaker status reporting."""
        status = get_circuit_breaker_status()

        for name in ["openai", "vectordb", "elasticsearch"]:
            assert name in status
            breaker_status = status[name]

            assert "state" in breaker_status
            assert "fail_counter" in breaker_status
            assert "fail_max" in breaker_status
            assert "timeout_duration" in breaker_status
            assert "is_available" in breaker_status

            # Initially, all should be closed and available
            assert breaker_status["state"] == "closed"
            assert breaker_status["is_available"] is True

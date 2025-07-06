"""
Prometheus metrics integration for comprehensive monitoring.
"""

import logging
import time
from typing import Dict, Any, Optional
from functools import wraps

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    CollectorRegistry,
    multiprocess,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
import psutil

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create custom registry for application metrics
REGISTRY = CollectorRegistry()

# Application Info
app_info = Info("app_info", "Application information", registry=REGISTRY)

# HTTP Metrics
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status_code"],
    registry=REGISTRY,
)

http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
    registry=REGISTRY,
)

# Search Metrics
search_requests_total = Counter(
    "search_requests_total",
    "Total search requests",
    ["search_type", "status"],
    registry=REGISTRY,
)

search_duration_seconds = Histogram(
    "search_duration_seconds",
    "Search operation duration in seconds",
    ["search_type"],
    registry=REGISTRY,
)

search_results_count = Histogram(
    "search_results_count",
    "Number of search results returned",
    ["search_type"],
    registry=REGISTRY,
)

# Document Processing Metrics
documents_processed_total = Counter(
    "documents_processed_total",
    "Total documents processed",
    ["status", "document_type"],
    registry=REGISTRY,
)

document_processing_duration_seconds = Histogram(
    "document_processing_duration_seconds",
    "Document processing duration in seconds",
    ["operation", "document_type"],
    registry=REGISTRY,
)

document_chunks_created_total = Counter(
    "document_chunks_created_total",
    "Total document chunks created",
    ["chunking_strategy"],
    registry=REGISTRY,
)

# Vector Database Metrics
vector_operations_total = Counter(
    "vector_operations_total",
    "Total vector database operations",
    ["operation", "provider", "status"],
    registry=REGISTRY,
)

vector_operation_duration_seconds = Histogram(
    "vector_operation_duration_seconds",
    "Vector database operation duration in seconds",
    ["operation", "provider"],
    registry=REGISTRY,
)

vector_index_size = Gauge(
    "vector_index_size",
    "Number of vectors in the index",
    ["provider", "collection"],
    registry=REGISTRY,
)

# Embedding Metrics
embeddings_generated_total = Counter(
    "embeddings_generated_total",
    "Total embeddings generated",
    ["provider", "model", "status"],
    registry=REGISTRY,
)

embedding_generation_duration_seconds = Histogram(
    "embedding_generation_duration_seconds",
    "Embedding generation duration in seconds",
    ["provider", "model"],
    registry=REGISTRY,
)

embedding_cache_hits_total = Counter(
    "embedding_cache_hits_total",
    "Total embedding cache hits",
    ["cache_type"],
    registry=REGISTRY,
)

# Rate Limiting Metrics
rate_limit_violations_total = Counter(
    "rate_limit_violations_total",
    "Total rate limit violations",
    ["client_type", "endpoint"],
    registry=REGISTRY,
)

active_rate_limited_clients = Gauge(
    "active_rate_limited_clients",
    "Number of currently rate limited clients",
    registry=REGISTRY,
)

# System Resource Metrics
system_cpu_usage_percent = Gauge(
    "system_cpu_usage_percent", "CPU usage percentage", registry=REGISTRY
)

system_memory_usage_percent = Gauge(
    "system_memory_usage_percent", "Memory usage percentage", registry=REGISTRY
)

system_disk_usage_percent = Gauge(
    "system_disk_usage_percent",
    "Disk usage percentage",
    ["mount_point"],
    registry=REGISTRY,
)

# Error Metrics
errors_total = Counter(
    "errors_total",
    "Total errors",
    ["error_type", "component", "severity"],
    registry=REGISTRY,
)

# Cache Metrics
cache_operations_total = Counter(
    "cache_operations_total",
    "Total cache operations",
    ["operation", "cache_type", "status"],
    registry=REGISTRY,
)

cache_size = Gauge(
    "cache_size", "Current cache size", ["cache_type"], registry=REGISTRY
)

# Connection Pool Metrics
database_connections_active = Gauge(
    "database_connections_active",
    "Active database connections",
    ["database_type"],
    registry=REGISTRY,
)

database_connections_idle = Gauge(
    "database_connections_idle",
    "Idle database connections",
    ["database_type"],
    registry=REGISTRY,
)


class MetricsCollector:
    """Centralized metrics collector for the application."""

    def __init__(self):
        self.start_time = time.time()
        self._update_app_info()

    def _update_app_info(self):
        """Update application information metrics."""
        app_info.info(
            {
                "version": settings.APP_VERSION,
                "name": settings.APP_NAME,
                "environment": "development" if settings.DEBUG else "production",
                "vector_db_provider": settings.VECTOR_DB_PROVIDER,
                "embedding_provider": getattr(settings, "EMBEDDING_PROVIDER", "openai"),
            }
        )

    def record_http_request(
        self, method: str, endpoint: str, status_code: int, duration: float
    ):
        """Record HTTP request metrics."""
        http_requests_total.labels(
            method=method, endpoint=endpoint, status_code=str(status_code)
        ).inc()

        http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
            duration
        )

    def record_search_operation(
        self, search_type: str, duration: float, result_count: int, success: bool = True
    ):
        """Record search operation metrics."""
        status = "success" if success else "error"

        search_requests_total.labels(search_type=search_type, status=status).inc()

        if success:
            search_duration_seconds.labels(search_type=search_type).observe(duration)
            search_results_count.labels(search_type=search_type).observe(result_count)

    def record_document_processing(
        self,
        operation: str,
        document_type: str,
        duration: float,
        chunks_created: int = 0,
        chunking_strategy: str = None,
        success: bool = True,
    ):
        """Record document processing metrics."""
        status = "success" if success else "error"

        documents_processed_total.labels(
            status=status, document_type=document_type
        ).inc()

        if success:
            document_processing_duration_seconds.labels(
                operation=operation, document_type=document_type
            ).observe(duration)

            if chunks_created > 0 and chunking_strategy:
                document_chunks_created_total.labels(
                    chunking_strategy=chunking_strategy
                ).inc(chunks_created)

    def record_vector_operation(
        self, operation: str, provider: str, duration: float, success: bool = True
    ):
        """Record vector database operation metrics."""
        status = "success" if success else "error"

        vector_operations_total.labels(
            operation=operation, provider=provider, status=status
        ).inc()

        if success:
            vector_operation_duration_seconds.labels(
                operation=operation, provider=provider
            ).observe(duration)

    def update_vector_index_size(self, provider: str, collection: str, size: int):
        """Update vector index size metric."""
        vector_index_size.labels(provider=provider, collection=collection).set(size)

    def record_embedding_generation(
        self,
        provider: str,
        model: str,
        duration: float,
        count: int = 1,
        success: bool = True,
    ):
        """Record embedding generation metrics."""
        status = "success" if success else "error"

        embeddings_generated_total.labels(
            provider=provider, model=model, status=status
        ).inc(count)

        if success:
            embedding_generation_duration_seconds.labels(
                provider=provider, model=model
            ).observe(duration)

    def record_cache_hit(self, cache_type: str):
        """Record cache hit."""
        embedding_cache_hits_total.labels(cache_type=cache_type).inc()

    def record_rate_limit_violation(self, client_type: str, endpoint: str):
        """Record rate limit violation."""
        rate_limit_violations_total.labels(
            client_type=client_type, endpoint=endpoint
        ).inc()

    def update_active_rate_limited_clients(self, count: int):
        """Update active rate limited clients count."""
        active_rate_limited_clients.set(count)

    def record_error(self, error_type: str, component: str, severity: str = "error"):
        """Record error occurrence."""
        errors_total.labels(
            error_type=error_type, component=component, severity=severity
        ).inc()

    def record_cache_operation(
        self, operation: str, cache_type: str, success: bool = True
    ):
        """Record cache operation."""
        status = "success" if success else "error"
        cache_operations_total.labels(
            operation=operation, cache_type=cache_type, status=status
        ).inc()

    def update_cache_size(self, cache_type: str, size: int):
        """Update cache size metric."""
        cache_size.labels(cache_type=cache_type).set(size)

    def update_database_connections(self, database_type: str, active: int, idle: int):
        """Update database connection metrics."""
        database_connections_active.labels(database_type=database_type).set(active)
        database_connections_idle.labels(database_type=database_type).set(idle)

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            system_cpu_usage_percent.set(cpu_percent)

            # Memory usage
            memory = psutil.virtual_memory()
            system_memory_usage_percent.set(memory.percent)

            # Disk usage for root partition
            disk = psutil.disk_usage("/")
            system_disk_usage_percent.labels(mount_point="/").set(
                (disk.used / disk.total) * 100
            )

        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics."""
        try:
            uptime = time.time() - self.start_time

            return {
                "uptime_seconds": uptime,
                "app_info": {
                    "name": settings.APP_NAME,
                    "version": settings.APP_VERSION,
                    "environment": "development" if settings.DEBUG else "production",
                },
                "system": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_percent": (
                        psutil.disk_usage("/").used / psutil.disk_usage("/").total
                    )
                    * 100,
                },
                "metrics_available": True,
            }
        except Exception as e:
            logger.error(f"Failed to get metrics summary: {e}")
            return {
                "uptime_seconds": time.time() - self.start_time,
                "error": str(e),
                "metrics_available": False,
            }


# Global metrics collector instance
_metrics_collector = None


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def generate_metrics() -> str:
    """Generate Prometheus metrics output."""
    try:
        # Update system metrics before generating output
        get_metrics_collector().update_system_metrics()

        # Handle multiprocess mode if needed
        if hasattr(multiprocess, "MultiProcessCollector"):
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            return generate_latest(registry)
        else:
            return generate_latest(REGISTRY)
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return f"# Error generating metrics: {e}\n"


def metrics_middleware(func):
    """Middleware decorator to automatically record HTTP metrics."""

    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start_time = time.time()
        status_code = 200

        try:
            response = await func(request, *args, **kwargs)
            if hasattr(response, "status_code"):
                status_code = response.status_code
            return response
        except Exception as e:
            status_code = 500
            get_metrics_collector().record_error(
                error_type=type(e).__name__, component="http", severity="error"
            )
            raise
        finally:
            duration = time.time() - start_time
            get_metrics_collector().record_http_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=duration,
            )

    return wrapper

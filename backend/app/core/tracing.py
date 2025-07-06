"""
OpenTelemetry tracing setup for the RAG platform.
"""

import logging
import os
from typing import Dict, Any, Optional
from functools import wraps

from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.semantic_conventions.resource import ResourceAttributes

from app.core.config import settings

logger = logging.getLogger(__name__)

# Global tracer and meter instances
tracer = None
meter = None
initialized = False


def init_tracing() -> None:
    """Initialize OpenTelemetry tracing and metrics."""
    global tracer, meter, initialized
    
    if initialized:
        logger.info("Tracing already initialized")
        return
    
    try:
        # Create resource with service information
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: "rag-platform",
            ResourceAttributes.SERVICE_VERSION: settings.APP_VERSION,
            ResourceAttributes.SERVICE_NAMESPACE: "enterprise-rag",
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: "development" if settings.DEBUG else "production",
        })
        
        # Set up tracing
        trace_provider = TracerProvider(resource=resource)
        trace.set_tracer_provider(trace_provider)
        
        # Configure exporters
        if settings.DEBUG:
            # Console exporter for development
            console_exporter = ConsoleSpanExporter()
            span_processor = BatchSpanProcessor(console_exporter)
            trace_provider.add_span_processor(span_processor)
            logger.info("Tracing configured with console exporter")
        
        # Jaeger exporter for production
        jaeger_endpoint = os.getenv("JAEGER_ENDPOINT", "http://localhost:14268/api/traces")
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                endpoint=jaeger_endpoint,
                collector_endpoint=jaeger_endpoint,
            )
            jaeger_processor = BatchSpanProcessor(jaeger_exporter)
            trace_provider.add_span_processor(jaeger_processor)
            logger.info(f"Tracing configured with Jaeger exporter: {jaeger_endpoint}")
        
        # Set up metrics
        metric_readers = []
        prometheus_port = int(os.getenv("PROMETHEUS_PORT", "8001"))
        if prometheus_port:
            prometheus_reader = PrometheusMetricReader(port=prometheus_port)
            metric_readers.append(prometheus_reader)
        
        metrics_provider = MeterProvider(
            resource=resource,
            metric_readers=metric_readers
        )
        metrics.set_meter_provider(metrics_provider)
        
        # Get tracer and meter instances
        tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)
        
        # Instrument libraries
        _instrument_libraries()
        
        initialized = True
        logger.info("OpenTelemetry tracing and metrics initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        # Create no-op tracer and meter as fallback
        tracer = trace.NoOpTracer()
        meter = metrics.NoOpMeter("fallback")


def _instrument_libraries() -> None:
    """Instrument common libraries for automatic tracing."""
    try:
        # Instrument HTTP clients
        HTTPXClientInstrumentor().instrument()
        
        # Instrument Redis if available
        try:
            RedisInstrumentor().instrument()
        except Exception as e:
            logger.warning(f"Failed to instrument Redis: {e}")
        
        # Instrument SQLAlchemy if available
        try:
            SQLAlchemyInstrumentor().instrument()
        except Exception as e:
            logger.warning(f"Failed to instrument SQLAlchemy: {e}")
        
        logger.info("Library instrumentation completed")
        
    except Exception as e:
        logger.error(f"Failed to instrument libraries: {e}")


def instrument_fastapi(app) -> None:
    """Instrument FastAPI application."""
    try:
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=trace.get_tracer_provider(),
            excluded_urls="health,metrics,docs,redoc,openapi.json"
        )
        logger.info("FastAPI instrumentation completed")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def get_tracer() -> trace.Tracer:
    """Get the global tracer instance."""
    global tracer
    if not initialized:
        init_tracing()
    return tracer or trace.NoOpTracer()


def get_meter() -> metrics.Meter:
    """Get the global meter instance."""
    global meter
    if not initialized:
        init_tracing()
    return meter or metrics.NoOpMeter("fallback")


def trace_function(operation_name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """Decorator to trace function execution."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            tracer_instance = get_tracer()
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer_instance.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)
                
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    if hasattr(func, '__await__'):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            tracer_instance = get_tracer()
            span_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            with tracer_instance.start_as_current_span(span_name) as span:
                if attributes:
                    span.set_attributes(attributes)
                
                # Add function metadata
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("success", True)
                    return result
                except Exception as e:
                    span.set_attribute("success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def trace_search_operation(search_type: str, query: str, limit: int = None):
    """Decorator specifically for search operations."""
    return trace_function(
        operation_name=f"search.{search_type}",
        attributes={
            "search.type": search_type,
            "search.query_length": len(query),
            "search.limit": limit,
        }
    )


def trace_document_processing(operation: str, document_id: str = None, filename: str = None):
    """Decorator specifically for document processing operations."""
    attributes = {"document.operation": operation}
    if document_id:
        attributes["document.id"] = document_id
    if filename:
        attributes["document.filename"] = filename
    
    return trace_function(
        operation_name=f"document.{operation}",
        attributes=attributes
    )


def trace_vector_operation(operation: str, provider: str = None, collection: str = None):
    """Decorator specifically for vector database operations."""
    attributes = {"vector.operation": operation}
    if provider:
        attributes["vector.provider"] = provider
    if collection:
        attributes["vector.collection"] = collection
    
    return trace_function(
        operation_name=f"vector.{operation}",
        attributes=attributes
    )


class TracingMetrics:
    """Metrics collection for tracing and performance monitoring."""
    
    def __init__(self):
        self.meter = get_meter()
        
        # Request counters
        self.request_counter = self.meter.create_counter(
            name="http_requests_total",
            description="Total number of HTTP requests",
            unit="1"
        )
        
        # Search metrics
        self.search_duration = self.meter.create_histogram(
            name="search_duration_seconds",
            description="Duration of search operations in seconds",
            unit="s"
        )
        
        self.search_counter = self.meter.create_counter(
            name="search_operations_total",
            description="Total number of search operations",
            unit="1"
        )
        
        # Document processing metrics
        self.document_processing_duration = self.meter.create_histogram(
            name="document_processing_duration_seconds",
            description="Duration of document processing in seconds",
            unit="s"
        )
        
        self.document_counter = self.meter.create_counter(
            name="documents_processed_total",
            description="Total number of documents processed",
            unit="1"
        )
        
        # Vector database metrics
        self.vector_operation_duration = self.meter.create_histogram(
            name="vector_operation_duration_seconds",
            description="Duration of vector database operations in seconds",
            unit="s"
        )
        
        self.vector_operation_counter = self.meter.create_counter(
            name="vector_operations_total",
            description="Total number of vector database operations",
            unit="1"
        )
        
        # Error metrics
        self.error_counter = self.meter.create_counter(
            name="errors_total",
            description="Total number of errors",
            unit="1"
        )
    
    def record_search(self, search_type: str, duration: float, success: bool = True):
        """Record search operation metrics."""
        labels = {"search_type": search_type, "success": str(success).lower()}
        self.search_counter.add(1, labels)
        self.search_duration.record(duration, labels)
    
    def record_document_processing(self, operation: str, duration: float, success: bool = True):
        """Record document processing metrics."""
        labels = {"operation": operation, "success": str(success).lower()}
        self.document_counter.add(1, labels)
        self.document_processing_duration.record(duration, labels)
    
    def record_vector_operation(self, operation: str, provider: str, duration: float, success: bool = True):
        """Record vector database operation metrics."""
        labels = {"operation": operation, "provider": provider, "success": str(success).lower()}
        self.vector_operation_counter.add(1, labels)
        self.vector_operation_duration.record(duration, labels)
    
    def record_error(self, error_type: str, operation: str = None):
        """Record error metrics."""
        labels = {"error_type": error_type}
        if operation:
            labels["operation"] = operation
        self.error_counter.add(1, labels)


# Global metrics instance
metrics_collector = None


def get_metrics_collector() -> TracingMetrics:
    """Get the global metrics collector instance."""
    global metrics_collector
    if metrics_collector is None:
        metrics_collector = TracingMetrics()
    return metrics_collector


def shutdown_tracing():
    """Shutdown tracing and cleanup resources."""
    global initialized
    try:
        # Flush any pending spans
        if trace.get_tracer_provider():
            for processor in trace.get_tracer_provider()._active_span_processor._span_processors:
                processor.force_flush()
                processor.shutdown()
        
        initialized = False
        logger.info("Tracing shutdown completed")
    except Exception as e:
        logger.error(f"Error during tracing shutdown: {e}")
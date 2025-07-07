"""
Main FastAPI application for the Enterprise RAG Platform.
"""

import logging
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from app.api.v1.api import api_router
from app.core.config import settings
from app.core.metrics import (
    CONTENT_TYPE_LATEST,
    generate_metrics,
    get_metrics_collector,
)
from app.core.rate_limiting import (
    custom_rate_limit_exceeded_handler,
    rate_limit_middleware,
)
from app.core.tracing import init_tracing, instrument_fastapi, shutdown_tracing
from app.services.search.search_manager import get_search_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Enterprise RAG Platform...")

    # Startup
    try:
        # Initialize OpenTelemetry tracing
        init_tracing()
        logger.info("OpenTelemetry tracing initialized")

        # Initialize search manager
        search_manager = await get_search_manager(settings)
        logger.info("Search manager initialized successfully")

        # Store in app state for access in endpoints
        app.state.search_manager = search_manager

        logger.info("Application startup complete")
        yield
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down Enterprise RAG Platform...")

        try:
            # Cleanup search manager
            if hasattr(app.state, "search_manager"):
                await app.state.search_manager.close()
                logger.info("Search manager closed successfully")

            # Shutdown tracing
            shutdown_tracing()
            logger.info("Tracing shutdown completed")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Enterprise-grade RAG platform with hybrid search capabilities",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# Add security middleware
if not settings.DEBUG:
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=settings.BACKEND_CORS_ORIGINS
    )

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting middleware
app.middleware("http")(rate_limit_middleware)

# Add rate limit exception handler
app.add_exception_handler(RateLimitExceeded, custom_rate_limit_exceeded_handler)


# Request timing and metrics middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header and record metrics."""
    start_time = time.time()
    status_code = 200

    try:
        response = await call_next(request)
        status_code = response.status_code
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)

        # Record metrics (skip for metrics endpoints to avoid recursion)
        if not request.url.path.startswith("/metrics"):
            get_metrics_collector().record_http_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=process_time,
            )

        return response
    except Exception as e:
        process_time = time.time() - start_time
        status_code = 500

        # Record error metrics
        if not request.url.path.startswith("/metrics"):
            get_metrics_collector().record_http_request(
                method=request.method,
                endpoint=request.url.path,
                status_code=status_code,
                duration=process_time,
            )
            get_metrics_collector().record_error(
                error_type=type(e).__name__,
                component="http_middleware",
                severity="error",
            )

        raise


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring."""
    start_time = time.time()

    # Log request
    logger.info(f"Request: {request.method} {request.url.path}")

    # Process request
    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"Response: {response.status_code} - {process_time:.3f}s - "
        f"{request.method} {request.url.path}"
    )

    return response


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    return JSONResponse(
        status_code=400,
        content={"error": "Bad Request", "detail": str(exc)},
    )


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Request, exc: FileNotFoundError):
    """Handle FileNotFoundError exceptions."""
    return JSONResponse(
        status_code=404,
        content={"error": "File Not Found", "detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )

    if settings.DEBUG:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": str(exc),
                "type": type(exc).__name__,
            },
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred",
            },
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Enterprise RAG Platform API",
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs" if settings.DEBUG else "Documentation disabled in production",
        "health": "/health",
        "api_base": settings.API_V1_STR,
    }


# Health check endpoint
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": time.time(),
    }


# Prometheus metrics endpoint
@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    from fastapi import Response

    metrics_data = generate_metrics()
    return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)


# Alternative metrics endpoint with JSON format
@app.get("/metrics/json")
async def metrics_json():
    """JSON metrics endpoint for monitoring dashboards."""
    collector = get_metrics_collector()
    return collector.get_metrics_summary()


# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

# Instrument FastAPI for tracing
instrument_fastapi(app)


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
# Test comment

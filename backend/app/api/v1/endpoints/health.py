"""
Health check endpoints.
"""

from fastapi import APIRouter
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check with service status."""
    # TODO: Add actual service health checks
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "services": {
            "vector_db": "pending",
            "embedding_service": "pending",
            "search_engine": "pending",
        },
    } 
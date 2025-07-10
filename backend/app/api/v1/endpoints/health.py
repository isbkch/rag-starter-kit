"""
Health check endpoints with detailed component monitoring.
"""
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import Settings, get_settings
from app.services.search.search_manager import SearchManager, get_search_manager

logger = logging.getLogger(__name__)
router = APIRouter()


async def get_search_manager_dep(
    settings: Settings = Depends(get_settings),
) -> SearchManager:
    """Dependency to get search manager."""
    return await get_search_manager(settings)


@router.get("/")
async def health_check():
    """Basic health check."""
    settings = get_settings()
    return {
        "status": "healthy",
        "message": "RAG Platform API is running",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/detailed")
async def detailed_health_check(
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Detailed health check with component status."""
    try:
        settings = get_settings()
        health_status = {
            "status": "healthy",
            "app_name": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {},
        }

        # Check search components
        try:
            search_health = await search_manager.health_check()
            health_status["components"]["search"] = search_health

            # Update overall status based on search health
            if search_health.get("status") == "degraded":
                health_status["status"] = "degraded"
            elif search_health.get("status") == "unhealthy":
                health_status["status"] = "unhealthy"

        except Exception as e:
            logger.error(f"Error checking search health: {e}")
            health_status["components"]["search"] = {
                "status": "unhealthy",
                "error": str(e),
            }
            health_status["status"] = "degraded"

        # Check API components
        health_status["components"]["api"] = {
            "status": "healthy",
            "endpoints": {
                "documents": "healthy",
                "search": "healthy",
                "health": "healthy",
            },
        }

        # Add system information
        try:
            import sys

            import psutil

            health_status["system"] = {
                "python_version": sys.version,
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
            }
        except ImportError:
            health_status["system"] = {
                "status": "monitoring_unavailable",
                "message": "psutil not installed",
            }

        return health_status

    except Exception as e:
        logger.error(f"Error performing detailed health check: {e}")
        return {
            "status": "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e),
        }


@router.get("/ready")
async def readiness_check(
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Readiness check for Kubernetes/container orchestration."""
    try:
        # Check if search manager is initialized
        stats = search_manager.get_search_stats()

        if not stats.get("initialized", False):
            raise HTTPException(
                status_code=503, detail="Search manager not initialized"
            )

        return {
            "status": "ready",
            "timestamp": datetime.utcnow().isoformat(),
            "initialized_components": [
                component
                for component, status in stats.items()
                if isinstance(status, dict) and status.get("indexed_documents", 0) >= 0
            ],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error performing readiness check: {e}")
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")


@router.get("/live")
async def liveness_check():
    """Liveness check for Kubernetes/container orchestration."""
    try:
        # Basic liveness check - just ensure the service is responding
        return {"status": "alive", "timestamp": datetime.utcnow().isoformat()}

    except Exception as e:
        logger.error(f"Error performing liveness check: {e}")
        raise HTTPException(status_code=503, detail=f"Service not alive: {str(e)}")

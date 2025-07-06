"""
Celery application configuration for background tasks.
"""

import os
from celery import Celery
from app.core.config import settings


def create_celery_app() -> Celery:
    """Create and configure Celery application."""

    celery_app = Celery(
        "rag_platform",
        broker=settings.CELERY_BROKER_URL,
        backend=settings.CELERY_RESULT_BACKEND,
        include=[
            "app.tasks.document_processing",
            "app.tasks.indexing",
            "app.tasks.maintenance",
        ],
    )

    # Celery configuration
    celery_app.conf.update(
        # Task routing
        task_routes={
            "app.tasks.document_processing.*": {"queue": "document_processing"},
            "app.tasks.indexing.*": {"queue": "indexing"},
            "app.tasks.maintenance.*": {"queue": "maintenance"},
        },
        # Task execution
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        # Task result settings
        result_expires=3600,  # 1 hour
        result_compression="gzip",
        # Worker settings
        worker_max_tasks_per_child=1000,
        worker_prefetch_multiplier=1,
        worker_concurrency=4,
        # Task execution settings
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_track_started=True,
        # Beat schedule for periodic tasks
        beat_schedule={
            "cleanup-old-results": {
                "task": "app.tasks.maintenance.cleanup_old_results",
                "schedule": 3600.0,  # Every hour
            },
            "update-search-stats": {
                "task": "app.tasks.maintenance.update_search_statistics",
                "schedule": 300.0,  # Every 5 minutes
            },
            "health-check": {
                "task": "app.tasks.maintenance.system_health_check",
                "schedule": 600.0,  # Every 10 minutes
            },
        },
    )

    return celery_app


# Create Celery app instance
celery = create_celery_app()


if __name__ == "__main__":
    celery.start()

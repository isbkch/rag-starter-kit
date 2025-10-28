"""
Database connection and session management.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

# Create database engine with explicit connection pooling configuration
engine_kwargs = {
    "pool_pre_ping": settings.DB_POOL_PRE_PING,  # Enable connection health checks
    "pool_recycle": settings.DB_POOL_RECYCLE,  # Recycle connections periodically
    "echo": settings.DEBUG,
}

# Add pool settings only for non-SQLite databases
if not settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs.update(
        {
            "pool_size": settings.DB_POOL_SIZE,  # Base connection pool size
            "max_overflow": settings.DB_MAX_OVERFLOW,  # Max additional connections
        }
    )

engine = create_engine(settings.DATABASE_URL, **engine_kwargs)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

"""
Core configuration module for the RAG platform.
"""

import os
from functools import lru_cache
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""

    # Application
    APP_NAME: str = "Enterprise RAG Platform"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days

    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # Redis
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")

    # Vector Database
    VECTOR_DB_PROVIDER: str = Field(default="chroma", env="VECTOR_DB_PROVIDER")
    VECTOR_DB_COLLECTION: str = Field(default="documents", env="VECTOR_DB_COLLECTION")

    # Pinecone
    PINECONE_API_KEY: Optional[str] = Field(default=None, env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: Optional[str] = Field(
        default=None, env="PINECONE_ENVIRONMENT"
    )
    PINECONE_INDEX_NAME: Optional[str] = Field(default=None, env="PINECONE_INDEX_NAME")

    # Weaviate
    WEAVIATE_URL: Optional[str] = Field(default=None, env="WEAVIATE_URL")
    WEAVIATE_API_KEY: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")

    # Chroma
    CHROMA_HOST: str = Field(default="localhost", env="CHROMA_HOST")
    CHROMA_PORT: int = Field(default=8000, env="CHROMA_PORT")

    # Elasticsearch
    ELASTICSEARCH_URL: str = Field(
        default="http://localhost:9200", env="ELASTICSEARCH_URL"
    )

    # OpenAI
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")

    # Embeddings
    EMBEDDING_PROVIDER: str = Field(default="openai", env="EMBEDDING_PROVIDER")
    EMBEDDING_MODEL: str = Field(
        default="text-embedding-ada-002", env="EMBEDDING_MODEL"
    )
    EMBEDDING_DIMENSIONS: int = Field(default=1536, env="EMBEDDING_DIMENSIONS")
    CACHE_EMBEDDINGS: bool = Field(default=True, env="CACHE_EMBEDDINGS")
    EMBEDDING_BATCH_SIZE: int = Field(default=100, env="EMBEDDING_BATCH_SIZE")

    # Document Processing
    MAX_FILE_SIZE: int = Field(default=50 * 1024 * 1024, env="MAX_FILE_SIZE")  # 50MB
    CHUNK_SIZE: int = Field(default=1000, env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(default=200, env="CHUNK_OVERLAP")
    CHUNKING_STRATEGY: str = Field(default="recursive", env="CHUNKING_STRATEGY")

    # Search
    MAX_SEARCH_RESULTS: int = Field(default=10, env="MAX_SEARCH_RESULTS")
    SIMILARITY_THRESHOLD: float = Field(default=0.7, env="SIMILARITY_THRESHOLD")

    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = Field(default=100, env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(default=3600, env="RATE_LIMIT_WINDOW")  # 1 hour

    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")

    # Celery Configuration
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379", env="CELERY_BROKER_URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379", env="CELERY_RESULT_BACKEND"
    )

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        env="BACKEND_CORS_ORIGINS",
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

"""
Vector database factory for creating the appropriate implementation.
"""

import logging
from typing import Optional

from .base import BaseVectorDB, VectorDBConfig
from .chroma_db import ChromaDB
from .pinecone_db import PineconeDB
from .weaviate_db import WeaviateDB
from app.core.config import settings

logger = logging.getLogger(__name__)


class VectorDBFactory:
    """Factory for creating vector database instances."""

    _instance = None
    _db_instance = None

    def __new__(cls):
        """Singleton pattern to ensure only one factory instance."""
        if cls._instance is None:
            cls._instance = super(VectorDBFactory, cls).__new__(cls)
        return cls._instance

    @classmethod
    def create_vector_db(
        cls,
        provider: Optional[str] = None,
        config: Optional[VectorDBConfig] = None,
    ) -> BaseVectorDB:
        """
        Create a vector database instance.

        Args:
            provider: Vector database provider ('chroma', 'pinecone', 'weaviate')
            config: Vector database configuration

        Returns:
            Vector database instance
        """
        if provider is None:
            provider = settings.VECTOR_DB_PROVIDER.lower()

        if config is None:
            config = VectorDBConfig(
                collection_name="documents",
                embedding_dimension=settings.EMBEDDING_DIMENSIONS,
                distance_metric="cosine",
            )

        provider = provider.lower()

        if provider == "chroma":
            return ChromaDB(config)
        elif provider == "pinecone":
            return PineconeDB(config)
        elif provider == "weaviate":
            return WeaviateDB(config)
        else:
            raise ValueError(f"Unsupported vector database provider: {provider}")

    @classmethod
    def get_vector_db(
        cls,
        provider: Optional[str] = None,
        config: Optional[VectorDBConfig] = None,
    ) -> BaseVectorDB:
        """
        Get a singleton vector database instance.

        Args:
            provider: Vector database provider
            config: Vector database configuration

        Returns:
            Vector database instance
        """
        if cls._db_instance is None:
            cls._db_instance = cls.create_vector_db(provider, config)

        return cls._db_instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (useful for testing)."""
        cls._db_instance = None

    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported vector database providers."""
        return ["chroma", "pinecone", "weaviate"]

    @classmethod
    def validate_provider_config(cls, provider: str) -> dict:
        """
        Validate configuration for a specific provider.

        Args:
            provider: Vector database provider

        Returns:
            Dictionary with validation results
        """
        provider = provider.lower()
        validation_result = {
            "provider": provider,
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "required_settings": [],
        }

        if provider == "chroma":
            validation_result["required_settings"] = ["CHROMA_HOST", "CHROMA_PORT"]

            if not settings.CHROMA_HOST:
                validation_result["errors"].append("CHROMA_HOST is required")
                validation_result["is_valid"] = False

            if not settings.CHROMA_PORT:
                validation_result["errors"].append("CHROMA_PORT is required")
                validation_result["is_valid"] = False

        elif provider == "pinecone":
            validation_result["required_settings"] = [
                "PINECONE_API_KEY",
                "PINECONE_ENVIRONMENT",
                "PINECONE_INDEX_NAME",
            ]

            if not settings.PINECONE_API_KEY:
                validation_result["errors"].append("PINECONE_API_KEY is required")
                validation_result["is_valid"] = False

            if not settings.PINECONE_ENVIRONMENT:
                validation_result["errors"].append("PINECONE_ENVIRONMENT is required")
                validation_result["is_valid"] = False

            if not settings.PINECONE_INDEX_NAME:
                validation_result["errors"].append("PINECONE_INDEX_NAME is required")
                validation_result["is_valid"] = False

        elif provider == "weaviate":
            validation_result["required_settings"] = ["WEAVIATE_URL"]

            if not settings.WEAVIATE_URL:
                validation_result["errors"].append("WEAVIATE_URL is required")
                validation_result["is_valid"] = False

            if not settings.WEAVIATE_API_KEY:
                validation_result["warnings"].append(
                    "WEAVIATE_API_KEY not set - using unauthenticated connection"
                )

        else:
            validation_result["errors"].append(f"Unsupported provider: {provider}")
            validation_result["is_valid"] = False

        return validation_result

    @classmethod
    async def test_connection(cls, provider: str) -> dict:
        """
        Test connection to a vector database provider.

        Args:
            provider: Vector database provider

        Returns:
            Dictionary with test results
        """
        test_result = {
            "provider": provider,
            "connection_successful": False,
            "error": None,
            "stats": {},
        }

        try:
            # Validate configuration first
            validation = cls.validate_provider_config(provider)
            if not validation["is_valid"]:
                test_result["error"] = f"Configuration invalid: {validation['errors']}"
                return test_result

            # Create and test connection
            db = cls.create_vector_db(provider)
            await db.connect()

            # Get health check
            health = await db.health_check()
            test_result["connection_successful"] = health["status"] == "healthy"
            test_result["stats"] = health.get("collection_stats", {})

            if not test_result["connection_successful"]:
                test_result["error"] = health.get("error", "Unknown connection error")

            await db.disconnect()

        except Exception as e:
            test_result["error"] = str(e)
            logger.error(f"Failed to test connection to {provider}: {e}")

        return test_result

    @classmethod
    def get_provider_info(cls, provider: str) -> dict:
        """
        Get information about a vector database provider.

        Args:
            provider: Vector database provider

        Returns:
            Dictionary with provider information
        """
        provider = provider.lower()

        provider_info = {
            "chroma": {
                "name": "ChromaDB",
                "description": "Open-source embedding database",
                "features": [
                    "Local and remote deployment",
                    "Built-in embedding functions",
                    "Metadata filtering",
                    "Collection management",
                ],
                "use_cases": [
                    "Development and testing",
                    "Small to medium scale deployments",
                    "On-premise deployments",
                ],
                "pricing": "Free (open-source)",
                "documentation": "https://docs.trychroma.com/",
            },
            "pinecone": {
                "name": "Pinecone",
                "description": "Managed vector database service",
                "features": [
                    "Fully managed service",
                    "High performance and scale",
                    "Metadata filtering",
                    "Serverless and pod-based options",
                ],
                "use_cases": [
                    "Production deployments",
                    "Large scale applications",
                    "Managed service preference",
                ],
                "pricing": "Usage-based pricing",
                "documentation": "https://docs.pinecone.io/",
            },
            "weaviate": {
                "name": "Weaviate",
                "description": "Open-source vector database with GraphQL API",
                "features": [
                    "GraphQL API",
                    "Built-in vectorization",
                    "Hybrid search capabilities",
                    "Multi-modal support",
                ],
                "use_cases": [
                    "Complex data relationships",
                    "Multi-modal applications",
                    "GraphQL preference",
                ],
                "pricing": "Free (open-source) + managed options",
                "documentation": "https://weaviate.io/developers/weaviate",
            },
        }

        return provider_info.get(
            provider,
            {
                "name": "Unknown",
                "description": "Unsupported provider",
                "error": f"Provider '{provider}' is not supported",
            },
        )

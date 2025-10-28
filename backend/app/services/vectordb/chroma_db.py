"""
ChromaDB vector database implementation.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from pybreaker import CircuitBreakerError

from app.core.circuit_breaker import async_circuit_breaker, vectordb_breaker
from app.core.config import settings

from .base import BaseVectorDB, VectorDBConfig, VectorSearchResult

logger = logging.getLogger(__name__)


class ChromaDB(BaseVectorDB):
    """ChromaDB implementation of vector database."""

    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Connect to ChromaDB."""
        try:
            # Create ChromaDB client
            chroma_settings = Settings(
                chroma_server_host=settings.CHROMA_HOST,
                chroma_server_http_port=settings.CHROMA_PORT,
                chroma_server_ssl_enabled=False,
                chroma_server_grpc_port=None,
                chroma_server_cors_allow_origins=["*"],
            )

            self._client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=chroma_settings,
            )

            # Test connection
            self._client.heartbeat()

            # Create or get collection
            await self._ensure_collection()

            logger.info(
                f"Connected to ChromaDB at "
                f"{settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
            )

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        self._client = None
        self._collection = None
        logger.info("Disconnected from ChromaDB")

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        collection_name = self.config.collection_name

        try:
            # Try to get existing collection
            self._collection = self._client.get_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction(),
            )
            logger.info(f"Using existing ChromaDB collection: {collection_name}")

        except Exception:
            # Collection doesn't exist, create it
            self._collection = self._client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction(),
                metadata={
                    "hnsw:space": self.config.distance_metric,
                    "description": "RAG Platform document chunks",
                },
            )
            logger.info(f"Created new ChromaDB collection: {collection_name}")

    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Create a new collection."""
        try:
            self._client.create_collection(
                name=collection_name,
                embedding_function=embedding_functions.DefaultEmbeddingFunction(),
                metadata=kwargs.get("metadata", {}),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            self._client.delete_collection(name=collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            self._client.get_collection(name=collection_name)
            return True
        except Exception:
            return False

    @async_circuit_breaker(vectordb_breaker)
    async def insert_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Insert vectors with metadata (protected by circuit breaker)."""
        try:
            collection = await self._get_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]

            # Extract documents (content) from metadata
            documents = [meta.get("content", "") for meta in metadata]

            # Clean metadata (remove content as it's stored separately)
            clean_metadata = []
            for meta in metadata:
                clean_meta = {k: v for k, v in meta.items() if k != "content"}
                clean_metadata.append(clean_meta)

            # Insert into ChromaDB
            collection.add(
                embeddings=vectors,
                documents=documents,
                metadatas=clean_metadata,
                ids=ids,
            )

            logger.info(f"Inserted {len(vectors)} vectors into ChromaDB")
            return ids

        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise

    @async_circuit_breaker(vectordb_breaker)
    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors (protected by circuit breaker)."""
        try:
            collection = await self._get_collection(collection_name)

            # Convert filters to ChromaDB format
            where_clause = None
            if filters:
                where_clause = self._convert_filters(filters)

            # Perform search
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=limit,
                where=where_clause,
                include=["metadatas", "documents", "distances"],
            )

            # Convert results to VectorSearchResult format
            search_results = []

            if results["ids"] and results["ids"][0]:
                for i, result_id in enumerate(results["ids"][0]):
                    score = (
                        1.0 - results["distances"][0][i]
                    )  # Convert distance to similarity
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    content = results["documents"][0][i] if results["documents"] else ""

                    search_results.append(
                        VectorSearchResult(
                            id=result_id,
                            score=score,
                            metadata=metadata,
                            content=content,
                        )
                    )

            return search_results

        except Exception as e:
            logger.error(f"Failed to search vectors: {e}")
            raise

    async def update_vectors(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Update existing vectors."""
        try:
            collection = await self._get_collection(collection_name)

            update_data = {"ids": ids}

            if vectors:
                update_data["embeddings"] = vectors

            if metadata:
                documents = [meta.get("content", "") for meta in metadata]
                clean_metadata = []
                for meta in metadata:
                    clean_meta = {k: v for k, v in meta.items() if k != "content"}
                    clean_metadata.append(clean_meta)

                update_data["documents"] = documents
                update_data["metadatas"] = clean_metadata

            collection.update(**update_data)
            return True

        except Exception as e:
            logger.error(f"Failed to update vectors: {e}")
            return False

    async def delete_vectors(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
    ) -> bool:
        """Delete vectors by IDs."""
        try:
            collection = await self._get_collection(collection_name)
            collection.delete(ids=ids)
            return True

        except Exception as e:
            logger.error(f"Failed to delete vectors: {e}")
            return False

    async def get_vector(
        self,
        vector_id: str,
        collection_name: Optional[str] = None,
    ) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        try:
            collection = await self._get_collection(collection_name)

            results = collection.get(
                ids=[vector_id],
                include=["metadatas", "documents", "embeddings"],
            )

            if results["ids"] and results["ids"][0]:
                metadata = results["metadatas"][0] if results["metadatas"] else {}
                content = results["documents"][0] if results["documents"] else ""
                embedding = results["embeddings"][0] if results["embeddings"] else None

                return VectorSearchResult(
                    id=vector_id,
                    score=1.0,  # Perfect match
                    metadata=metadata,
                    content=content,
                    embedding=embedding,
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get vector {vector_id}: {e}")
            return None

    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            collection = await self._get_collection(collection_name)

            # Get collection count
            count = collection.count()

            return {
                "name": collection.name,
                "count": count,
                "embedding_dimension": self.config.embedding_dimension,
                "distance_metric": self.config.distance_metric,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    async def _get_collection(self, collection_name: Optional[str] = None):
        """Get collection instance."""
        if collection_name and collection_name != self.config.collection_name:
            # Get a different collection
            return self._client.get_collection(name=collection_name)

        if self._collection is None:
            await self._ensure_collection()

        return self._collection

    def _convert_filters(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """Convert filters to ChromaDB where clause format."""
        # ChromaDB uses a specific format for filters
        # This is a simplified conversion - can be extended
        where_clause = {}

        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                where_clause[key] = {"$eq": value}
            elif isinstance(value, list):
                where_clause[key] = {"$in": value}
            elif isinstance(value, dict):
                # Handle operators like {"$gt": 10}
                where_clause[key] = value

        return where_clause

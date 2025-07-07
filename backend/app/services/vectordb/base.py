"""
Base vector database interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.models.documents import DocumentChunk


@dataclass
class VectorSearchResult:
    """Vector search result."""

    id: str
    score: float
    metadata: Dict[str, Any]
    content: str
    embedding: Optional[List[float]] = None


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    collection_name: str = "documents"
    embedding_dimension: int = 1536
    distance_metric: str = "cosine"  # cosine, euclidean, dot_product
    index_params: Optional[Dict[str, Any]] = None
    connection_params: Optional[Dict[str, Any]] = None


class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""

    def __init__(self, config: VectorDBConfig):
        self.config = config
        self._client = None

    @abstractmethod
    async def connect(self) -> None:
        """Connect to the vector database."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from the vector database."""
        pass

    @abstractmethod
    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Create a new collection/index."""
        pass

    @abstractmethod
    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection/index."""
        pass

    @abstractmethod
    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        pass

    @abstractmethod
    async def insert_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Insert vectors with metadata."""
        pass

    @abstractmethod
    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    async def update_vectors(
        self,
        ids: List[str],
        vectors: Optional[List[List[float]]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Update existing vectors."""
        pass

    @abstractmethod
    async def delete_vectors(
        self,
        ids: List[str],
        collection_name: Optional[str] = None,
    ) -> bool:
        """Delete vectors by IDs."""
        pass

    @abstractmethod
    async def get_vector(
        self,
        vector_id: str,
        collection_name: Optional[str] = None,
    ) -> Optional[VectorSearchResult]:
        """Get a specific vector by ID."""
        pass

    @abstractmethod
    async def get_collection_stats(
        self,
        collection_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get collection statistics."""
        pass

    # Helper methods

    async def insert_document_chunks(
        self,
        chunks: List[DocumentChunk],
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Insert document chunks into the vector database."""
        if not chunks:
            return []

        # Extract vectors and metadata
        vectors = []
        metadata = []
        ids = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(f"Chunk {chunk.id} has no embedding")

            vectors.append(chunk.embedding)
            metadata.append(
                {
                    "chunk_id": chunk.id,
                    "document_id": chunk.document_id,
                    "content": chunk.content,
                    "chunk_index": chunk.chunk_index,
                    "start_char": chunk.start_char,
                    "end_char": chunk.end_char,
                    "created_at": chunk.created_at.isoformat(),
                    **chunk.metadata,
                }
            )
            ids.append(chunk.id)

        return await self.insert_vectors(
            vectors=vectors,
            metadata=metadata,
            ids=ids,
            collection_name=collection_name,
        )

    async def search_by_text(
        self,
        query_embedding: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar text chunks."""
        return await self.search_vectors(
            query_vector=query_embedding,
            limit=limit,
            filters=filters,
            collection_name=collection_name,
        )

    async def delete_document_chunks(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Delete all chunks for a document."""
        # This is a default implementation - can be overridden for efficiency
        filters = {"document_id": document_id}

        # First, search for all chunks of this document
        results = await self.search_vectors(
            query_vector=[0.0] * self.config.embedding_dimension,
            limit=10000,  # Large limit to get all chunks
            filters=filters,
            collection_name=collection_name,
        )

        if not results:
            return True

        # Delete all found chunks
        ids = [result.id for result in results]
        return await self.delete_vectors(ids, collection_name)

    def _get_collection_name(self, collection_name: Optional[str] = None) -> str:
        """Get the collection name to use."""
        return collection_name or self.config.collection_name

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector database connection."""
        try:
            if self._client is None:
                await self.connect()

            # Try to get collection stats as a health check
            stats = await self.get_collection_stats()

            return {
                "status": "healthy",
                "connected": True,
                "collection_stats": stats,
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "connected": False,
                "error": str(e),
            }

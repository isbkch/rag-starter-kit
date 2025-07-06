"""
Pinecone vector database implementation.
"""

import logging
from typing import List, Dict, Any, Optional
import uuid

import pinecone
from pinecone import Pinecone, ServerlessSpec

from .base import BaseVectorDB, VectorSearchResult, VectorDBConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class PineconeDB(BaseVectorDB):
    """Pinecone implementation of vector database."""

    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self._index = None

    async def connect(self) -> None:
        """Connect to Pinecone."""
        try:
            if not settings.PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY not configured")

            # Initialize Pinecone client
            self._client = Pinecone(api_key=settings.PINECONE_API_KEY)

            # Get or create index
            await self._ensure_index()

            logger.info(f"Connected to Pinecone index: {settings.PINECONE_INDEX_NAME}")

        except Exception as e:
            logger.error(f"Failed to connect to Pinecone: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Pinecone."""
        self._client = None
        self._index = None
        logger.info("Disconnected from Pinecone")

    async def _ensure_index(self) -> None:
        """Ensure the index exists."""
        index_name = settings.PINECONE_INDEX_NAME

        try:
            # Check if index exists
            if index_name not in self._client.list_indexes().names():
                # Create index
                self._client.create_index(
                    name=index_name,
                    dimension=self.config.embedding_dimension,
                    metric=self.config.distance_metric,
                    spec=ServerlessSpec(
                        cloud="aws", region=settings.PINECONE_ENVIRONMENT or "us-east-1"
                    ),
                )
                logger.info(f"Created Pinecone index: {index_name}")

            # Connect to index
            self._index = self._client.Index(index_name)
            logger.info(f"Connected to Pinecone index: {index_name}")

        except Exception as e:
            logger.error(f"Failed to ensure Pinecone index: {e}")
            raise

    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Create a new index (collection in Pinecone terms)."""
        try:
            dimension = kwargs.get("dimension", self.config.embedding_dimension)
            metric = kwargs.get("metric", self.config.distance_metric)

            self._client.create_index(
                name=collection_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud="aws", region=settings.PINECONE_ENVIRONMENT or "us-east-1"
                ),
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create Pinecone index {collection_name}: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete an index."""
        try:
            self._client.delete_index(collection_name)
            return True

        except Exception as e:
            logger.error(f"Failed to delete Pinecone index {collection_name}: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if index exists."""
        try:
            return collection_name in self._client.list_indexes().names()
        except Exception:
            return False

    async def insert_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> List[str]:
        """Insert vectors with metadata."""
        try:
            index = await self._get_index(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]

            # Prepare vectors for upsert
            vectors_to_upsert = []
            for i, vector in enumerate(vectors):
                vectors_to_upsert.append(
                    {
                        "id": ids[i],
                        "values": vector,
                        "metadata": metadata[i],
                    }
                )

            # Batch upsert (Pinecone recommends batches of 100)
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i : i + batch_size]
                index.upsert(vectors=batch)

            logger.info(f"Inserted {len(vectors)} vectors into Pinecone")
            return ids

        except Exception as e:
            logger.error(f"Failed to insert vectors: {e}")
            raise

    async def search_vectors(
        self,
        query_vector: List[float],
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[VectorSearchResult]:
        """Search for similar vectors."""
        try:
            index = await self._get_index(collection_name)

            # Perform search
            results = index.query(
                vector=query_vector,
                top_k=limit,
                filter=filters,
                include_metadata=True,
                include_values=False,
            )

            # Convert results to VectorSearchResult format
            search_results = []

            for match in results["matches"]:
                search_results.append(
                    VectorSearchResult(
                        id=match["id"],
                        score=match["score"],
                        metadata=match.get("metadata", {}),
                        content=match.get("metadata", {}).get("content", ""),
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
            index = await self._get_index(collection_name)

            # Prepare update data
            vectors_to_update = []
            for i, vector_id in enumerate(ids):
                update_data = {"id": vector_id}

                if vectors and i < len(vectors):
                    update_data["values"] = vectors[i]

                if metadata and i < len(metadata):
                    update_data["metadata"] = metadata[i]

                vectors_to_update.append(update_data)

            # Batch update
            batch_size = 100
            for i in range(0, len(vectors_to_update), batch_size):
                batch = vectors_to_update[i : i + batch_size]
                index.upsert(vectors=batch)

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
            index = await self._get_index(collection_name)

            # Batch delete
            batch_size = 1000  # Pinecone allows up to 1000 IDs per delete
            for i in range(0, len(ids), batch_size):
                batch = ids[i : i + batch_size]
                index.delete(ids=batch)

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
            index = await self._get_index(collection_name)

            results = index.fetch(ids=[vector_id])

            if vector_id in results["vectors"]:
                vector_data = results["vectors"][vector_id]

                return VectorSearchResult(
                    id=vector_id,
                    score=1.0,  # Perfect match
                    metadata=vector_data.get("metadata", {}),
                    content=vector_data.get("metadata", {}).get("content", ""),
                    embedding=vector_data.get("values"),
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
            index = await self._get_index(collection_name)

            # Get index stats
            stats = index.describe_index_stats()

            return {
                "name": settings.PINECONE_INDEX_NAME,
                "dimension": stats["dimension"],
                "index_fullness": stats["index_fullness"],
                "total_vector_count": stats["total_vector_count"],
                "namespaces": stats.get("namespaces", {}),
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    async def _get_index(self, collection_name: Optional[str] = None):
        """Get index instance."""
        if collection_name and collection_name != settings.PINECONE_INDEX_NAME:
            # Get a different index
            return self._client.Index(collection_name)

        if self._index is None:
            await self._ensure_index()

        return self._index

    async def delete_document_chunks(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Delete all chunks for a document."""
        try:
            index = await self._get_index(collection_name)

            # Delete by metadata filter
            index.delete(filter={"document_id": document_id})
            return True

        except Exception as e:
            logger.error(f"Failed to delete document chunks for {document_id}: {e}")
            return False

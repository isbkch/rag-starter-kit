"""
Weaviate vector database implementation.
"""

import logging
from typing import List, Dict, Any, Optional
import uuid

import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import Filter

from .base import BaseVectorDB, VectorSearchResult, VectorDBConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class WeaviateDB(BaseVectorDB):
    """Weaviate implementation of vector database."""

    def __init__(self, config: VectorDBConfig):
        super().__init__(config)
        self._client = None
        self._collection = None

    async def connect(self) -> None:
        """Connect to Weaviate."""
        try:
            # Configure Weaviate client
            auth_config = None
            if settings.WEAVIATE_API_KEY:
                auth_config = weaviate.auth.AuthApiKey(
                    api_key=settings.WEAVIATE_API_KEY
                )

            self._client = weaviate.connect_to_local(
                host=settings.WEAVIATE_URL.replace("http://", "").replace(
                    "https://", ""
                ),
                port=8080,
                grpc_port=50051,
                auth=auth_config,
            )

            # Test connection
            if not self._client.is_ready():
                raise Exception("Weaviate client is not ready")

            # Create or get collection
            await self._ensure_collection()

            logger.info(f"Connected to Weaviate at {settings.WEAVIATE_URL}")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from Weaviate."""
        if self._client:
            self._client.close()
        self._client = None
        self._collection = None
        logger.info("Disconnected from Weaviate")

    async def _ensure_collection(self) -> None:
        """Ensure the collection exists."""
        collection_name = (
            self.config.collection_name.capitalize()
        )  # Weaviate requires capitalized names

        try:
            # Check if collection exists
            if self._client.collections.exists(collection_name):
                self._collection = self._client.collections.get(collection_name)
                logger.info(f"Using existing Weaviate collection: {collection_name}")
            else:
                # Create collection
                self._collection = self._client.collections.create(
                    name=collection_name,
                    properties=[
                        Property(name="content", data_type=DataType.TEXT),
                        Property(name="document_id", data_type=DataType.TEXT),
                        Property(name="chunk_id", data_type=DataType.TEXT),
                        Property(name="chunk_index", data_type=DataType.INT),
                        Property(name="start_char", data_type=DataType.INT),
                        Property(name="end_char", data_type=DataType.INT),
                        Property(name="created_at", data_type=DataType.TEXT),
                        Property(name="metadata", data_type=DataType.OBJECT),
                    ],
                    vectorizer_config=Configure.Vectorizer.none(),  # We provide our own vectors
                )
                logger.info(f"Created new Weaviate collection: {collection_name}")

        except Exception as e:
            logger.error(f"Failed to ensure Weaviate collection: {e}")
            raise

    async def create_collection(self, collection_name: str, **kwargs) -> bool:
        """Create a new collection."""
        try:
            collection_name = collection_name.capitalize()

            properties = kwargs.get(
                "properties",
                [
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="metadata", data_type=DataType.OBJECT),
                ],
            )

            self._client.collections.create(
                name=collection_name,
                properties=properties,
                vectorizer_config=Configure.Vectorizer.none(),
            )
            return True

        except Exception as e:
            logger.error(f"Failed to create Weaviate collection {collection_name}: {e}")
            return False

    async def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection."""
        try:
            collection_name = collection_name.capitalize()
            self._client.collections.delete(collection_name)
            return True

        except Exception as e:
            logger.error(f"Failed to delete Weaviate collection {collection_name}: {e}")
            return False

    async def collection_exists(self, collection_name: str) -> bool:
        """Check if collection exists."""
        try:
            collection_name = collection_name.capitalize()
            return self._client.collections.exists(collection_name)
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
            collection = await self._get_collection(collection_name)

            # Generate IDs if not provided
            if ids is None:
                ids = [str(uuid.uuid4()) for _ in vectors]

            # Prepare objects for insertion
            objects_to_insert = []
            for i, vector in enumerate(vectors):
                # Flatten metadata for Weaviate
                obj_data = {
                    "content": metadata[i].get("content", ""),
                    "document_id": metadata[i].get("document_id", ""),
                    "chunk_id": metadata[i].get("chunk_id", ""),
                    "chunk_index": metadata[i].get("chunk_index", 0),
                    "start_char": metadata[i].get("start_char", 0),
                    "end_char": metadata[i].get("end_char", 0),
                    "created_at": metadata[i].get("created_at", ""),
                    "metadata": {
                        k: v
                        for k, v in metadata[i].items()
                        if k
                        not in [
                            "content",
                            "document_id",
                            "chunk_id",
                            "chunk_index",
                            "start_char",
                            "end_char",
                            "created_at",
                        ]
                    },
                }

                objects_to_insert.append(
                    {
                        "uuid": ids[i],
                        "properties": obj_data,
                        "vector": vector,
                    }
                )

            # Batch insert
            batch_size = 100
            for i in range(0, len(objects_to_insert), batch_size):
                batch = objects_to_insert[i : i + batch_size]

                with collection.batch.dynamic() as batch_context:
                    for obj in batch:
                        batch_context.add_object(
                            properties=obj["properties"],
                            vector=obj["vector"],
                            uuid=obj["uuid"],
                        )

            logger.info(f"Inserted {len(vectors)} vectors into Weaviate")
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
            collection = await self._get_collection(collection_name)

            # Build query
            query = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                return_metadata=["distance"],
            )

            # Apply filters if provided
            if filters:
                where_filter = self._convert_filters(filters)
                query = query.where(where_filter)

            # Execute search
            results = query.objects

            # Convert results to VectorSearchResult format
            search_results = []

            for result in results:
                # Convert distance to similarity score
                distance = result.metadata.distance if result.metadata else 0
                score = 1.0 / (1.0 + distance)  # Convert distance to similarity

                # Reconstruct metadata
                metadata = dict(result.properties.get("metadata", {}))
                metadata.update(
                    {
                        "document_id": result.properties.get("document_id", ""),
                        "chunk_id": result.properties.get("chunk_id", ""),
                        "chunk_index": result.properties.get("chunk_index", 0),
                        "start_char": result.properties.get("start_char", 0),
                        "end_char": result.properties.get("end_char", 0),
                        "created_at": result.properties.get("created_at", ""),
                    }
                )

                search_results.append(
                    VectorSearchResult(
                        id=str(result.uuid),
                        score=score,
                        metadata=metadata,
                        content=result.properties.get("content", ""),
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

            for i, vector_id in enumerate(ids):
                update_data = {}

                if metadata and i < len(metadata):
                    # Flatten metadata for Weaviate
                    meta = metadata[i]
                    update_data.update(
                        {
                            "content": meta.get("content", ""),
                            "document_id": meta.get("document_id", ""),
                            "chunk_id": meta.get("chunk_id", ""),
                            "chunk_index": meta.get("chunk_index", 0),
                            "start_char": meta.get("start_char", 0),
                            "end_char": meta.get("end_char", 0),
                            "created_at": meta.get("created_at", ""),
                            "metadata": {
                                k: v
                                for k, v in meta.items()
                                if k
                                not in [
                                    "content",
                                    "document_id",
                                    "chunk_id",
                                    "chunk_index",
                                    "start_char",
                                    "end_char",
                                    "created_at",
                                ]
                            },
                        }
                    )

                # Update object
                collection.data.update(
                    uuid=vector_id,
                    properties=update_data,
                    vector=vectors[i] if vectors and i < len(vectors) else None,
                )

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

            for vector_id in ids:
                collection.data.delete_by_id(vector_id)

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

            result = collection.query.fetch_object_by_id(
                vector_id,
                include_vector=True,
            )

            if result:
                # Reconstruct metadata
                metadata = dict(result.properties.get("metadata", {}))
                metadata.update(
                    {
                        "document_id": result.properties.get("document_id", ""),
                        "chunk_id": result.properties.get("chunk_id", ""),
                        "chunk_index": result.properties.get("chunk_index", 0),
                        "start_char": result.properties.get("start_char", 0),
                        "end_char": result.properties.get("end_char", 0),
                        "created_at": result.properties.get("created_at", ""),
                    }
                )

                return VectorSearchResult(
                    id=str(result.uuid),
                    score=1.0,  # Perfect match
                    metadata=metadata,
                    content=result.properties.get("content", ""),
                    embedding=result.vector,
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

            # Get object count
            result = collection.aggregate.over_all(total_count=True)

            return {
                "name": collection.name,
                "count": result.total_count,
                "embedding_dimension": self.config.embedding_dimension,
                "distance_metric": self.config.distance_metric,
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    async def _get_collection(self, collection_name: Optional[str] = None):
        """Get collection instance."""
        if collection_name:
            collection_name = collection_name.capitalize()
            return self._client.collections.get(collection_name)

        if self._collection is None:
            await self._ensure_collection()

        return self._collection

    def _convert_filters(self, filters: Dict[str, Any]) -> Filter:
        """Convert filters to Weaviate Filter format."""
        # This is a simplified conversion - can be extended
        filter_conditions = []

        for key, value in filters.items():
            if isinstance(value, (str, int, float, bool)):
                filter_conditions.append(Filter.by_property(key).equal(value))
            elif isinstance(value, list):
                # Handle 'in' operations
                or_conditions = [Filter.by_property(key).equal(v) for v in value]
                if len(or_conditions) == 1:
                    filter_conditions.append(or_conditions[0])
                else:
                    # Combine with OR
                    combined = or_conditions[0]
                    for condition in or_conditions[1:]:
                        combined = combined | condition
                    filter_conditions.append(combined)

        # Combine all conditions with AND
        if len(filter_conditions) == 1:
            return filter_conditions[0]
        elif len(filter_conditions) > 1:
            combined = filter_conditions[0]
            for condition in filter_conditions[1:]:
                combined = combined & condition
            return combined

        return None

    async def delete_document_chunks(
        self,
        document_id: str,
        collection_name: Optional[str] = None,
    ) -> bool:
        """Delete all chunks for a document."""
        try:
            collection = await self._get_collection(collection_name)

            # Delete by document_id filter
            collection.data.delete_many(
                where=Filter.by_property("document_id").equal(document_id)
            )
            return True

        except Exception as e:
            logger.error(f"Failed to delete document chunks for {document_id}: {e}")
            return False

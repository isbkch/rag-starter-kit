"""
Search manager that coordinates all search engines and provides unified interface.
"""
import logging
from enum import Enum
from typing import Any, Dict, List, Optional

from app.core.cache import get_cache_service
from app.core.config import Settings
from app.core.metrics import get_metrics_collector
from app.core.tracing import trace_search_operation
from app.models.search import SearchResponse
from app.services.search.hybrid_search import HybridSearchEngine
from app.services.search.keyword_search import KeywordSearchEngine
from app.services.search.vector_search import VectorSearchEngine
from app.services.vectordb.factory import VectorDBFactory

logger = logging.getLogger(__name__)


class SearchType(str, Enum):
    """Available search types."""

    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchManager:
    """Manages all search engines and provides unified search interface."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._vector_search = None
        self._keyword_search = None
        self._hybrid_search = None
        self._embedding_service = None
        self._vector_db = None
        self._initialized = False

    async def initialize(self):
        """Initialize all search engines."""
        try:
            if self._initialized:
                return

            logger.info("Initializing search manager...")

            # Initialize vector database
            self._vector_db = VectorDBFactory.create_vector_db(
                provider=self.settings.VECTOR_DB_PROVIDER
            )

            # Initialize embedding service
            from app.services.search.embedding_service import get_embedding_service

            api_key = self.settings.OPENAI_API_KEY
            if not api_key:
                raise ValueError("OpenAI API key is required for embedding service")

            self._embedding_service = await get_embedding_service(
                provider=self.settings.EMBEDDING_PROVIDER,
                model_name=self.settings.EMBEDDING_MODEL,
                api_key=api_key,
                cache_embeddings=self.settings.CACHE_EMBEDDINGS,
                batch_size=self.settings.EMBEDDING_BATCH_SIZE,
            )

            # Initialize vector search engine
            self._vector_search = VectorSearchEngine(
                vector_db=self._vector_db,
                embedding_service=self._embedding_service,
                collection_name=self.settings.VECTOR_DB_COLLECTION,
            )

            # Initialize keyword search engine
            self._keyword_search = KeywordSearchEngine()

            # Initialize hybrid search engine
            self._hybrid_search = HybridSearchEngine(
                vector_search_engine=self._vector_search,
                keyword_search_engine=self._keyword_search,
                vector_weight=0.7,
                keyword_weight=0.3,
                fusion_method="rrf",
            )

            self._initialized = True
            logger.info("Search manager initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing search manager: {e}")
            raise

    @trace_search_operation(search_type="dynamic", query="", limit=10)
    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> SearchResponse:
        """Perform search using specified search type with caching."""
        import time

        start_time = time.time()
        metrics = get_metrics_collector()

        if not self._initialized:
            await self.initialize()

        # Try to get cached results first
        try:
            cache_service = await get_cache_service()
            cached_results = await cache_service.get_search_results(
                query=query,
                search_type=search_type.value,
                limit=limit,
                filters=filters,
                min_score=kwargs.get("min_score", 0.0),
            )

            if cached_results:
                logger.debug(f"Cache hit for search: {search_type.value} query")
                # Convert cached dict back to SearchResponse
                cached_response = SearchResponse(**cached_results)

                # Record cache hit metrics
                duration = time.time() - start_time
                metrics.record_search_operation(
                    search_type=search_type.value,
                    duration=duration,
                    result_count=len(cached_response.results),
                    success=True,
                )
                return cached_response
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}")
            # Continue with regular search

        try:
            result = None
            if search_type == SearchType.VECTOR:
                result = await self._search_vector(query, limit, filters, **kwargs)
            elif search_type == SearchType.KEYWORD:
                result = await self._search_keyword(query, limit, **kwargs)
            elif search_type == SearchType.HYBRID:
                result = await self._search_hybrid(query, limit, filters, **kwargs)
            else:
                raise ValueError(f"Unknown search type: {search_type}")

            # Cache the search results
            try:
                cache_service = await get_cache_service()
                # Convert SearchResponse to dict for caching
                result_dict = result.dict()
                await cache_service.set_search_results(
                    query=query,
                    search_type=search_type.value,
                    limit=limit,
                    results=result_dict,
                    filters=filters,
                    min_score=kwargs.get("min_score", 0.0),
                )
                logger.debug(f"Cached search results for {search_type.value} query")
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")

            # Record successful search metrics
            duration = time.time() - start_time
            metrics.record_search_operation(
                search_type=search_type.value,
                duration=duration,
                result_count=len(result.results) if result.results else 0,
                success=True,
            )
            return result

        except Exception as e:
            logger.error(f"Error performing {search_type} search: {e}")

            # Record failed search metrics
            duration = time.time() - start_time
            metrics.record_search_operation(
                search_type=search_type.value,
                duration=duration,
                result_count=0,
                success=False,
            )
            metrics.record_error(
                error_type=type(e).__name__,
                component=f"search.{search_type.value}",
                severity="error",
            )

            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=duration,
                search_type=search_type,
                error=str(e),
            )

    async def _search_vector(
        self, query: str, limit: int, filters: Optional[Dict[str, Any]], **kwargs
    ) -> SearchResponse:
        """Perform vector search."""
        min_score = kwargs.get("min_score", 0.0)
        return await self._vector_search.search(
            query=query, limit=limit, min_score=min_score, filters=filters
        )

    async def _search_keyword(self, query: str, limit: int, **kwargs) -> SearchResponse:
        """Perform keyword search."""
        min_score = kwargs.get("min_score", 0.0)
        return await self._keyword_search.search(
            query=query, limit=limit, min_score=min_score
        )

    async def _search_hybrid(
        self, query: str, limit: int, filters: Optional[Dict[str, Any]], **kwargs
    ) -> SearchResponse:
        """Perform hybrid search."""
        vector_limit = kwargs.get("vector_limit", limit)
        keyword_limit = kwargs.get("keyword_limit", limit)

        return await self._hybrid_search.search(
            query=query,
            limit=limit,
            filters=filters,
            min_vector_score=kwargs.get("min_vector_score", 0.0),
            min_keyword_score=kwargs.get("min_keyword_score", 0.0),
            vector_limit=vector_limit,
            keyword_limit=keyword_limit,
        )

    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents in all search engines."""
        if not self._initialized:
            await self.initialize()

        try:
            logger.info(f"Indexing {len(documents)} documents...")

            # Index in vector search (this will also handle vector DB storage)
            if self._vector_search and hasattr(self._vector_search, "index_documents"):
                await self._vector_search.index_documents(documents)
            else:
                logger.warning("Vector search engine not available for indexing")

            # Index in keyword search
            if self._keyword_search and hasattr(
                self._keyword_search, "index_documents"
            ):
                result = self._keyword_search.index_documents(documents)
                # Handle both sync and async keyword search implementations
                if result is not None and hasattr(result, "__await__"):
                    await result
            else:
                logger.warning("Keyword search engine not available for indexing")

            logger.info("Documents indexed successfully in all search engines")

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            raise

    def update_hybrid_weights(self, vector_weight: float, keyword_weight: float):
        """Update hybrid search weights."""
        if self._hybrid_search:
            self._hybrid_search.update_weights(vector_weight, keyword_weight)

    def get_search_stats(self) -> Dict[str, Any]:
        """Get statistics from all search engines."""
        stats = {
            "initialized": self._initialized,
            "vector_db_provider": self.settings.VECTOR_DB_PROVIDER,
            "embedding_provider": self.settings.EMBEDDING_PROVIDER,
        }

        if self._initialized:
            if self._vector_search:
                stats["vector_search"] = self._vector_search.get_stats()
            if self._keyword_search:
                stats["keyword_search"] = self._keyword_search.get_stats()
            if self._hybrid_search:
                stats["hybrid_search"] = self._hybrid_search.get_stats()

        return stats

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all search components."""
        health = {"status": "healthy", "components": {}}

        try:
            # Check vector database
            if self._vector_db:
                try:
                    await self._vector_db.health_check()
                    health["components"]["vector_db"] = "healthy"
                except Exception as e:
                    health["components"]["vector_db"] = f"unhealthy: {e}"
                    health["status"] = "degraded"

            # Check embedding service
            if self._embedding_service:
                try:
                    # Try to get embeddings for a test query
                    test_embedding = await self._embedding_service.get_embeddings(
                        ["test"]
                    )
                    if test_embedding and len(test_embedding) > 0:
                        health["components"]["embedding_service"] = "healthy"
                    else:
                        health["components"][
                            "embedding_service"
                        ] = "unhealthy: no embeddings returned"
                        health["status"] = "degraded"
                except Exception as e:
                    health["components"]["embedding_service"] = f"unhealthy: {e}"
                    health["status"] = "degraded"

            # Check search engines
            health["components"]["vector_search"] = (
                "healthy" if self._vector_search else "not_initialized"
            )
            health["components"]["keyword_search"] = (
                "healthy" if self._keyword_search else "not_initialized"
            )
            health["components"]["hybrid_search"] = (
                "healthy" if self._hybrid_search else "not_initialized"
            )

        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)

        return health

    async def close(self):
        """Close all search engine connections."""
        try:
            if self._vector_search:
                await self._vector_search.cleanup()
                logger.info("Vector search engine cleaned up")

            if self._vector_db:
                await self._vector_db.disconnect()
                logger.info("Vector database disconnected")

            if self._embedding_service and hasattr(self._embedding_service, "close"):
                await self._embedding_service.close()
                logger.info("Embedding service closed")

            if self._hybrid_search and hasattr(self._hybrid_search, "executor"):
                self._hybrid_search.executor.shutdown(wait=True)
                logger.info("Hybrid search executor shut down")

            # Reset initialization state
            self._initialized = False
            self._vector_search = None
            self._keyword_search = None
            self._hybrid_search = None
            self._embedding_service = None
            self._vector_db = None

            logger.info("Search manager closed successfully")

        except Exception as e:
            logger.error(f"Error closing search manager: {e}")
            raise


# Global search manager instance
search_manager = None


async def get_search_manager(settings: Settings = None) -> SearchManager:
    """Get or create global search manager instance."""
    global search_manager

    if search_manager is None:
        if settings is None:
            from app.core.config import get_settings

            settings = get_settings()

        search_manager = SearchManager(settings)
        await search_manager.initialize()
    elif not search_manager._initialized:
        # Re-initialize if closed
        await search_manager.initialize()

    return search_manager


async def close_search_manager():
    """Close and cleanup global search manager."""
    global search_manager

    if search_manager is not None:
        await search_manager.close()
        search_manager = None
        logger.info("Global search manager closed and reset")

"""
Search manager that coordinates all search engines and provides unified interface.
"""
import logging
from typing import Dict, Any, Optional, List
from enum import Enum

from app.core.config import Settings
from app.models.search import SearchResponse
from app.services.vectordb.factory import VectorDBFactory
from app.services.search.embedding_service import EmbeddingService
from app.services.search.vector_search import VectorSearchEngine
from app.services.search.keyword_search import KeywordSearchEngine
from app.services.search.hybrid_search import HybridSearchEngine

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
                batch_size=self.settings.EMBEDDING_BATCH_SIZE
            )
            
            # Initialize vector search engine
            self._vector_search = VectorSearchEngine(
                vector_db=self._vector_db,
                embedding_service=self._embedding_service,
                collection_name=self.settings.VECTOR_DB_COLLECTION
            )
            
            # Initialize keyword search engine
            self._keyword_search = KeywordSearchEngine()
            
            # Initialize hybrid search engine
            self._hybrid_search = HybridSearchEngine(
                vector_search_engine=self._vector_search,
                keyword_search_engine=self._keyword_search,
                vector_weight=0.7,
                keyword_weight=0.3,
                fusion_method="rrf"
            )
            
            self._initialized = True
            logger.info("Search manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing search manager: {e}")
            raise
    
    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> SearchResponse:
        """Perform search using specified search type."""
        if not self._initialized:
            await self.initialize()
        
        try:
            if search_type == SearchType.VECTOR:
                return await self._search_vector(query, limit, filters, **kwargs)
            elif search_type == SearchType.KEYWORD:
                return await self._search_keyword(query, limit, **kwargs)
            elif search_type == SearchType.HYBRID:
                return await self._search_hybrid(query, limit, filters, **kwargs)
            else:
                raise ValueError(f"Unknown search type: {search_type}")
                
        except Exception as e:
            logger.error(f"Error performing {search_type} search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0,
                search_type=search_type,
                error=str(e)
            )
    
    async def _search_vector(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        **kwargs
    ) -> SearchResponse:
        """Perform vector search."""
        min_score = kwargs.get('min_score', 0.0)
        return self._vector_search.search(
            query=query,
            limit=limit,
            min_score=min_score,
            filters=filters
        )
    
    async def _search_keyword(
        self,
        query: str,
        limit: int,
        **kwargs
    ) -> SearchResponse:
        """Perform keyword search."""
        min_score = kwargs.get('min_score', 0.0)
        return self._keyword_search.search(
            query=query,
            limit=limit,
            min_score=min_score
        )
    
    async def _search_hybrid(
        self,
        query: str,
        limit: int,
        filters: Optional[Dict[str, Any]],
        **kwargs
    ) -> SearchResponse:
        """Perform hybrid search."""
        return await self._hybrid_search.search(
            query=query,
            limit=limit,
            filters=filters,
            min_vector_score=kwargs.get('min_vector_score', 0.0),
            min_keyword_score=kwargs.get('min_keyword_score', 0.0),
            vector_limit=kwargs.get('vector_limit'),
            keyword_limit=kwargs.get('keyword_limit')
        )
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents in all search engines."""
        if not self._initialized:
            await self.initialize()
        
        try:
            logger.info(f"Indexing {len(documents)} documents...")
            
            # Index in vector search (this will also handle vector DB storage)
            await self._vector_search.index_documents(documents)
            
            # Index in keyword search
            self._keyword_search.index_documents(documents)
            
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
            "embedding_provider": self.settings.EMBEDDING_PROVIDER
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
        health = {
            "status": "healthy",
            "components": {}
        }
        
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
                    test_embedding = await self._embedding_service.get_embeddings(["test"])
                    if test_embedding and len(test_embedding) > 0:
                        health["components"]["embedding_service"] = "healthy"
                    else:
                        health["components"]["embedding_service"] = "unhealthy: no embeddings returned"
                        health["status"] = "degraded"
                except Exception as e:
                    health["components"]["embedding_service"] = f"unhealthy: {e}"
                    health["status"] = "degraded"
            
            # Check search engines
            health["components"]["vector_search"] = "healthy" if self._vector_search else "not_initialized"
            health["components"]["keyword_search"] = "healthy" if self._keyword_search else "not_initialized"
            health["components"]["hybrid_search"] = "healthy" if self._hybrid_search else "not_initialized"
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health
    
    async def close(self):
        """Close all search engine connections."""
        try:
            if self._vector_db:
                await self._vector_db.close()
            
            if self._hybrid_search and hasattr(self._hybrid_search, 'executor'):
                self._hybrid_search.executor.shutdown(wait=True)
            
            logger.info("Search manager closed successfully")
            
        except Exception as e:
            logger.error(f"Error closing search manager: {e}")

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
    
    return search_manager

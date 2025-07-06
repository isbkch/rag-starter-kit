"""
Embedding service implementation with multiple provider support and caching.
"""

import logging
import hashlib
import asyncio
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

import openai
import numpy as np
from sentence_transformers import SentenceTransformer
import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""
    
    @abstractmethod
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this provider."""
        pass


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if "ada-002" in model else 1536  # Default for most models
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI API."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        return self._dimension


class SentenceTransformerProvider(EmbeddingProvider):
    """Sentence Transformers embedding provider for local embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            self._dimension = self.model.get_sentence_embedding_dimension()
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using sentence transformers."""
        try:
            self._load_model()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                self.model.encode, 
                texts
            )
            
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error getting Sentence Transformer embeddings: {e}")
            raise
    
    def get_embedding_dimension(self) -> int:
        if self._dimension is None:
            self._load_model()
        return self._dimension


class EmbeddingCache:
    """Redis-based embedding cache."""
    
    def __init__(self, redis_url: str, ttl: int = 86400 * 7):  # 7 days default TTL
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis_client = None
    
    async def connect(self):
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(
                self.redis_url,
                decode_responses=False
            )
            await self.redis_client.ping()
            logger.info("Connected to Redis for embedding cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
    
    def _get_cache_key(self, text: str, model: str) -> str:
        """Generate cache key for text and model."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return f"embedding:{model}:{text_hash}"
    
    async def get(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        if not self.redis_client:
            return None
        
        try:
            key = self._get_cache_key(text, model)
            cached_data = await self.redis_client.get(key)
            
            if cached_data:
                # Deserialize numpy array
                embedding = np.frombuffer(cached_data, dtype=np.float32).tolist()
                logger.debug(f"Cache hit for text hash: {key}")
                return embedding
            
            return None
        except Exception as e:
            logger.warning(f"Error getting from cache: {e}")
            return None
    
    async def set(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache."""
        if not self.redis_client:
            return
        
        try:
            key = self._get_cache_key(text, model)
            # Serialize as numpy array for efficient storage
            embedding_bytes = np.array(embedding, dtype=np.float32).tobytes()
            
            await self.redis_client.setex(
                key,
                self.ttl,
                embedding_bytes
            )
            logger.debug(f"Cached embedding for text hash: {key}")
        except Exception as e:
            logger.warning(f"Error setting cache: {e}")
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.redis_client:
            return {"cache_enabled": False}
        
        try:
            info = await self.redis_client.info("memory")
            keyspace = await self.redis_client.info("keyspace")
            
            # Count embedding keys
            embedding_keys = 0
            async for key in self.redis_client.scan_iter(match="embedding:*"):
                embedding_keys += 1
            
            return {
                "cache_enabled": True,
                "memory_usage_mb": round(info.get("used_memory", 0) / 1024 / 1024, 2),
                "embedding_keys": embedding_keys,
                "keyspace_info": keyspace
            }
        except Exception as e:
            logger.warning(f"Error getting cache stats: {e}")
            return {"cache_enabled": True, "error": str(e)}
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class EmbeddingService:
    """Main embedding service with provider abstraction and caching."""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "text-embedding-ada-002",
        api_key: Optional[str] = None,
        cache_embeddings: bool = True,
        batch_size: int = 100
    ):
        self.provider = provider
        self.model_name = model_name
        self.cache_embeddings = cache_embeddings
        self.batch_size = batch_size
        
        # Initialize provider
        if provider == "openai":
            if not api_key:
                raise ValueError("OpenAI API key is required for OpenAI provider")
            self.embedding_provider = OpenAIEmbeddingProvider(api_key, model_name)
        elif provider == "sentence_transformers":
            self.embedding_provider = SentenceTransformerProvider(model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
        
        # Initialize cache
        self.cache = None
        if cache_embeddings:
            self.cache = EmbeddingCache(settings.REDIS_URL)
    
    async def initialize(self):
        """Initialize the embedding service."""
        if self.cache:
            await self.cache.connect()
        logger.info(f"Embedding service initialized with provider: {self.provider}")
    
    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a list of texts with caching."""
        if not texts:
            return []
        
        embeddings = []
        texts_to_compute = []
        indices_to_compute = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if self.cache:
                cached_embedding = await self.cache.get(text, self.model_name)
                if cached_embedding:
                    embeddings.append(cached_embedding)
                    continue
            
            # Mark for computation
            embeddings.append(None)  # Placeholder
            texts_to_compute.append(text)
            indices_to_compute.append(i)
        
        # Compute embeddings for uncached texts
        if texts_to_compute:
            logger.info(f"Computing embeddings for {len(texts_to_compute)} texts")
            
            # Process in batches
            computed_embeddings = []
            for i in range(0, len(texts_to_compute), self.batch_size):
                batch_texts = texts_to_compute[i:i + self.batch_size]
                batch_embeddings = await self.embedding_provider.get_embeddings(batch_texts)
                computed_embeddings.extend(batch_embeddings)
                
                # Cache the computed embeddings
                if self.cache:
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        await self.cache.set(text, self.model_name, embedding)
            
            # Fill in the computed embeddings
            for i, embedding in zip(indices_to_compute, computed_embeddings):
                embeddings[i] = embedding
        
        return embeddings
    
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text."""
        embeddings = await self.get_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.embedding_provider.get_embedding_dimension()
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the embedding service."""
        health = {
            "provider": self.provider,
            "model": self.model_name,
            "dimension": self.get_embedding_dimension(),
            "cache_enabled": self.cache is not None
        }
        
        try:
            # Test with a simple text
            test_embedding = await self.get_embedding("test")
            health["status"] = "healthy"
            health["test_embedding_length"] = len(test_embedding)
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        # Get cache stats if available
        if self.cache:
            cache_stats = await self.cache.get_stats()
            health["cache_stats"] = cache_stats
        
        return health
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get embedding service statistics."""
        stats = {
            "provider": self.provider,
            "model": self.model_name,
            "dimension": self.get_embedding_dimension(),
            "batch_size": self.batch_size,
            "cache_enabled": self.cache is not None
        }
        
        if self.cache:
            cache_stats = await self.cache.get_stats()
            stats.update(cache_stats)
        
        return stats
    
    async def close(self):
        """Close the embedding service."""
        if self.cache:
            await self.cache.close()
        logger.info("Embedding service closed")


# Global embedding service instance
_embedding_service = None


async def get_embedding_service(
    provider: str = None,
    model_name: str = None,
    api_key: str = None,
    cache_embeddings: bool = None,
    batch_size: int = None
) -> EmbeddingService:
    """Get or create global embedding service instance."""
    global _embedding_service
    
    if _embedding_service is None:
        # Use settings defaults if not provided
        provider = provider or settings.EMBEDDING_PROVIDER
        model_name = model_name or settings.EMBEDDING_MODEL
        api_key = api_key or settings.OPENAI_API_KEY
        cache_embeddings = cache_embeddings if cache_embeddings is not None else settings.CACHE_EMBEDDINGS
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE
        
        _embedding_service = EmbeddingService(
            provider=provider,
            model_name=model_name,
            api_key=api_key,
            cache_embeddings=cache_embeddings,
            batch_size=batch_size
        )
        
        await _embedding_service.initialize()
    
    return _embedding_service

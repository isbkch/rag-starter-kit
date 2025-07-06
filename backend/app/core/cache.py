"""
Redis-based caching service for embeddings, search results, and other data.
"""

import json
import logging
import hashlib
import pickle
from typing import Any, Optional, List, Dict, Union
from datetime import datetime, timedelta
import asyncio

import redis.asyncio as redis
from redis.asyncio import Redis

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheKeyBuilder:
    """Build consistent cache keys for different data types."""

    @staticmethod
    def embedding_key(text: str, model: str, provider: str = "openai") -> str:
        """Build cache key for embeddings."""
        # Use hash of text for consistent keys and to handle long texts
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"embedding:{provider}:{model}:{text_hash}"

    @staticmethod
    def search_key(
        query: str,
        search_type: str,
        limit: int,
        filters: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> str:
        """Build cache key for search results."""
        # Create a consistent hash of search parameters
        search_params = {
            "query": query,
            "search_type": search_type,
            "limit": limit,
            "min_score": min_score,
            "filters": filters or {},
        }
        params_str = json.dumps(search_params, sort_keys=True)
        params_hash = hashlib.sha256(params_str.encode()).hexdigest()[:16]
        return f"search:{search_type}:{params_hash}"

    @staticmethod
    def document_key(document_id: str, operation: str = "metadata") -> str:
        """Build cache key for document data."""
        return f"document:{operation}:{document_id}"

    @staticmethod
    def chunk_key(chunk_id: str) -> str:
        """Build cache key for document chunks."""
        return f"chunk:{chunk_id}"

    @staticmethod
    def vector_stats_key(provider: str, collection: str) -> str:
        """Build cache key for vector database statistics."""
        return f"vector_stats:{provider}:{collection}"


class RedisCache:
    """Redis-based cache implementation with async support."""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._client: Optional[Redis] = None
        self._connected = False

        # Default TTL values (in seconds)
        self.default_ttl = {
            "embedding": 24 * 60 * 60,  # 24 hours
            "search": 5 * 60,  # 5 minutes
            "document": 60 * 60,  # 1 hour
            "chunk": 2 * 60 * 60,  # 2 hours
            "vector_stats": 10 * 60,  # 10 minutes
            "default": 30 * 60,  # 30 minutes
        }

    async def connect(self) -> bool:
        """Connect to Redis."""
        try:
            self._client = redis.from_url(
                self.redis_url,
                decode_responses=False,  # Keep binary for pickle
                retry_on_timeout=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )

            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info("Connected to Redis cache")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to Redis cache: {e}")
            self._connected = False
            self._client = None
            return False

    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            try:
                await self._client.close()
                logger.info("Disconnected from Redis cache")
            except Exception as e:
                logger.warning(f"Error disconnecting from Redis: {e}")
            finally:
                self._client = None
                self._connected = False

    async def _ensure_connected(self):
        """Ensure Redis connection is active."""
        if not self._connected or not self._client:
            await self.connect()

        if not self._connected:
            raise ConnectionError("Unable to connect to Redis cache")

    def _get_ttl(self, cache_type: str) -> int:
        """Get TTL for cache type."""
        return self.default_ttl.get(cache_type, self.default_ttl["default"])

    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        cache_type: str = "default",
    ) -> bool:
        """Set value in cache."""
        try:
            await self._ensure_connected()

            # Serialize value
            if isinstance(value, (dict, list)):
                serialized_value = json.dumps(value).encode()
            elif isinstance(value, str):
                serialized_value = value.encode()
            elif isinstance(value, (int, float)):
                serialized_value = str(value).encode()
            else:
                # Use pickle for complex objects
                serialized_value = pickle.dumps(value)

            # Set TTL
            if ttl is None:
                ttl = self._get_ttl(cache_type)

            # Store in Redis
            await self._client.setex(key, ttl, serialized_value)
            logger.debug(f"Cached value for key: {key} (TTL: {ttl}s)")
            return True

        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False

    async def get(
        self, key: str, default: Any = None, deserialize_json: bool = True
    ) -> Any:
        """Get value from cache."""
        try:
            await self._ensure_connected()

            value = await self._client.get(key)
            if value is None:
                return default

            # Deserialize value
            if deserialize_json:
                try:
                    return json.loads(value.decode())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass

            # Try string decoding
            try:
                return value.decode()
            except UnicodeDecodeError:
                pass

            # Try pickle
            try:
                return pickle.loads(value)
            except (pickle.PickleError, EOFError):
                pass

            # Return raw bytes as fallback
            return value

        except Exception as e:
            logger.error(f"Failed to get cache key {key}: {e}")
            return default

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        try:
            await self._ensure_connected()
            result = await self._client.delete(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete cache key {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            await self._ensure_connected()
            result = await self._client.exists(key)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to check cache key {key}: {e}")
            return False

    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration for existing key."""
        try:
            await self._ensure_connected()
            result = await self._client.expire(key, ttl)
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to set expiration for key {key}: {e}")
            return False

    async def get_ttl(self, key: str) -> int:
        """Get remaining TTL for key."""
        try:
            await self._ensure_connected()
            return await self._client.ttl(key)
        except Exception as e:
            logger.error(f"Failed to get TTL for key {key}: {e}")
            return -1

    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        try:
            await self._ensure_connected()

            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                deleted = await self._client.delete(*keys)
                logger.info(f"Cleared {deleted} keys matching pattern: {pattern}")
                return deleted

            return 0

        except Exception as e:
            logger.error(f"Failed to clear pattern {pattern}: {e}")
            return 0

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            await self._ensure_connected()
            info = await self._client.info()

            return {
                "connected": self._connected,
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_commands_processed": info.get("total_commands_processed"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "hit_rate": self._calculate_hit_rate(
                    info.get("keyspace_hits", 0), info.get("keyspace_misses", 0)
                ),
            }
        except Exception as e:
            return {"connected": False, "error": str(e)}

    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate."""
        total = hits + misses
        return (hits / total * 100) if total > 0 else 0.0


class CacheService:
    """High-level cache service with specialized methods."""

    def __init__(self, redis_cache: RedisCache = None):
        self.cache = redis_cache or RedisCache()
        self.key_builder = CacheKeyBuilder()

    async def connect(self):
        """Connect to cache backend."""
        return await self.cache.connect()

    async def disconnect(self):
        """Disconnect from cache backend."""
        await self.cache.disconnect()

    # Embedding cache methods
    async def get_embedding(
        self, text: str, model: str, provider: str = "openai"
    ) -> Optional[List[float]]:
        """Get cached embedding."""
        key = self.key_builder.embedding_key(text, model, provider)
        embedding = await self.cache.get(key, deserialize_json=True)

        if embedding:
            logger.debug(f"Cache hit for embedding: {key}")
            # Record cache hit metric
            from app.core.metrics import get_metrics_collector

            get_metrics_collector().record_cache_operation(
                "get", "embedding", success=True
            )
            return embedding

        return None

    async def set_embedding(
        self, text: str, model: str, embedding: List[float], provider: str = "openai"
    ) -> bool:
        """Cache embedding."""
        key = self.key_builder.embedding_key(text, model, provider)
        success = await self.cache.set(key, embedding, cache_type="embedding")

        if success:
            logger.debug(f"Cached embedding: {key}")
            # Record cache operation metric
            from app.core.metrics import get_metrics_collector

            get_metrics_collector().record_cache_operation(
                "set", "embedding", success=True
            )

        return success

    # Search cache methods
    async def get_search_results(
        self,
        query: str,
        search_type: str,
        limit: int,
        filters: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> Optional[Dict]:
        """Get cached search results."""
        key = self.key_builder.search_key(query, search_type, limit, filters, min_score)
        results = await self.cache.get(key, deserialize_json=True)

        if results:
            logger.debug(f"Cache hit for search: {key}")
            # Record cache hit metric
            from app.core.metrics import get_metrics_collector

            get_metrics_collector().record_cache_operation(
                "get", "search", success=True
            )

        return results

    async def set_search_results(
        self,
        query: str,
        search_type: str,
        limit: int,
        results: Dict,
        filters: Optional[Dict] = None,
        min_score: float = 0.0,
    ) -> bool:
        """Cache search results."""
        key = self.key_builder.search_key(query, search_type, limit, filters, min_score)
        success = await self.cache.set(key, results, cache_type="search")

        if success:
            logger.debug(f"Cached search results: {key}")
            # Record cache operation metric
            from app.core.metrics import get_metrics_collector

            get_metrics_collector().record_cache_operation(
                "set", "search", success=True
            )

        return success

    # Document cache methods
    async def get_document_metadata(self, document_id: str) -> Optional[Dict]:
        """Get cached document metadata."""
        key = self.key_builder.document_key(document_id, "metadata")
        return await self.cache.get(key, deserialize_json=True)

    async def set_document_metadata(self, document_id: str, metadata: Dict) -> bool:
        """Cache document metadata."""
        key = self.key_builder.document_key(document_id, "metadata")
        return await self.cache.set(key, metadata, cache_type="document")

    # Vector stats cache methods
    async def get_vector_stats(self, provider: str, collection: str) -> Optional[Dict]:
        """Get cached vector database stats."""
        key = self.key_builder.vector_stats_key(provider, collection)
        return await self.cache.get(key, deserialize_json=True)

    async def set_vector_stats(
        self, provider: str, collection: str, stats: Dict
    ) -> bool:
        """Cache vector database stats."""
        key = self.key_builder.vector_stats_key(provider, collection)
        return await self.cache.set(key, stats, cache_type="vector_stats")

    # Bulk operations
    async def invalidate_search_cache(self):
        """Invalidate all search results cache."""
        pattern = "search:*"
        count = await self.cache.clear_pattern(pattern)
        logger.info(f"Invalidated {count} search cache entries")
        return count

    async def invalidate_document_cache(self, document_id: str = None):
        """Invalidate document cache (all or specific document)."""
        if document_id:
            pattern = f"document:*:{document_id}"
        else:
            pattern = "document:*"

        count = await self.cache.clear_pattern(pattern)
        logger.info(f"Invalidated {count} document cache entries")
        return count

    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        redis_stats = await self.cache.get_stats()

        # Add application-specific stats
        stats = {
            "redis": redis_stats,
            "key_patterns": {
                "embedding": await self._count_keys("embedding:*"),
                "search": await self._count_keys("search:*"),
                "document": await self._count_keys("document:*"),
                "chunk": await self._count_keys("chunk:*"),
                "vector_stats": await self._count_keys("vector_stats:*"),
            },
        }

        return stats

    async def _count_keys(self, pattern: str) -> int:
        """Count keys matching pattern."""
        try:
            await self.cache._ensure_connected()
            count = 0
            async for _ in self.cache._client.scan_iter(match=pattern):
                count += 1
            return count
        except Exception as e:
            logger.error(f"Failed to count keys for pattern {pattern}: {e}")
            return 0


# Global cache service instance
_cache_service = None


async def get_cache_service() -> CacheService:
    """Get the global cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
        await _cache_service.connect()
    return _cache_service


async def close_cache_service():
    """Close the global cache service."""
    global _cache_service
    if _cache_service:
        await _cache_service.disconnect()
        _cache_service = None

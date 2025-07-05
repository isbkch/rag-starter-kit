"""
Embedding service for generating vector embeddings.
"""

import logging
from typing import List, Optional, Dict, Any
import asyncio
from abc import ABC, abstractmethod

import openai
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Base class for embedding providers."""
    
    @abstractmethod
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        pass
    
    @abstractmethod
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        pass


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """OpenAI embedding provider."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-ada-002"):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.model = model
        self._dimension = 1536 if model == "text-embedding-ada-002" else 1536
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            # OpenAI API supports batch embedding
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
            )
            
            # Extract embeddings from response
            embeddings = [item.embedding for item in response.data]
            
            logger.debug(f"Generated {len(embeddings)} embeddings using OpenAI")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings with OpenAI: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self._dimension


class SentenceTransformerProvider(BaseEmbeddingProvider):
    """Sentence Transformer embedding provider."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._dimension = None
    
    def _load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
            # Get dimension from model
            self._dimension = self.model.get_sentence_embedding_dimension()
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            self._load_model()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, self.model.encode, texts
            )
            
            # Convert to list of lists
            embeddings_list = [embedding.tolist() for embedding in embeddings]
            
            logger.debug(f"Generated {len(embeddings_list)} embeddings using SentenceTransformer")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings with SentenceTransformer: {e}")
            raise
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        if self._dimension is None:
            self._load_model()
        return self._dimension


class EmbeddingService:
    """Main embedding service with multiple provider support."""
    
    def __init__(self, provider: str = "openai", **kwargs):
        self.provider_name = provider.lower()
        self.provider = self._create_provider(provider, **kwargs)
        self._cache = {}  # Simple in-memory cache
        self.cache_enabled = kwargs.get('cache_enabled', True)
        self.max_cache_size = kwargs.get('max_cache_size', 1000)
    
    def _create_provider(self, provider: str, **kwargs) -> BaseEmbeddingProvider:
        """Create embedding provider instance."""
        provider = provider.lower()
        
        if provider == "openai":
            api_key = kwargs.get('api_key', settings.OPENAI_API_KEY)
            model = kwargs.get('model', settings.EMBEDDING_MODEL)
            
            if not api_key:
                raise ValueError("OpenAI API key is required")
            
            return OpenAIEmbeddingProvider(api_key=api_key, model=model)
        
        elif provider == "sentence-transformer":
            model_name = kwargs.get('model_name', 'all-MiniLM-L6-v2')
            return SentenceTransformerProvider(model_name=model_name)
        
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        if not text.strip():
            raise ValueError("Text cannot be empty")
        
        # Check cache first
        if self.cache_enabled and text in self._cache:
            logger.debug("Retrieved embedding from cache")
            return self._cache[text]
        
        # Generate embedding
        embedding = await self.provider.embed_text(text)
        
        # Cache the result
        if self.cache_enabled:
            self._update_cache(text, embedding)
        
        return embedding
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")
        
        # Check cache for existing embeddings
        cached_embeddings = {}
        texts_to_embed = []
        
        if self.cache_enabled:
            for text in valid_texts:
                if text in self._cache:
                    cached_embeddings[text] = self._cache[text]
                else:
                    texts_to_embed.append(text)
        else:
            texts_to_embed = valid_texts
        
        # Generate embeddings for uncached texts
        new_embeddings = {}
        if texts_to_embed:
            embeddings = await self.provider.embed_texts(texts_to_embed)
            new_embeddings = dict(zip(texts_to_embed, embeddings))
            
            # Update cache
            if self.cache_enabled:
                for text, embedding in new_embeddings.items():
                    self._update_cache(text, embedding)
        
        # Combine cached and new embeddings in original order
        result = []
        for text in valid_texts:
            if text in cached_embeddings:
                result.append(cached_embeddings[text])
            else:
                result.append(new_embeddings[text])
        
        return result
    
    async def embed_document_chunks(self, chunks: List[str]) -> List[List[float]]:
        """Generate embeddings for document chunks with batching."""
        if not chunks:
            return []
        
        # Process in batches to avoid API limits
        batch_size = 100  # Adjust based on provider limits
        all_embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = await self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            
            # Add small delay between batches to avoid rate limits
            if i + batch_size < len(chunks):
                await asyncio.sleep(0.1)
        
        return all_embeddings
    
    def _update_cache(self, text: str, embedding: List[float]):
        """Update the embedding cache."""
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[text] = embedding
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self._cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'cache_size': len(self._cache),
            'max_cache_size': self.max_cache_size,
            'cache_enabled': self.cache_enabled,
            'hit_rate': getattr(self, '_cache_hits', 0) / max(getattr(self, '_cache_requests', 1), 1),
        }
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        return self.provider.get_embedding_dimension()
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider."""
        return {
            'provider': self.provider_name,
            'dimension': self.get_embedding_dimension(),
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self._cache),
        }
    
    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            # Convert to numpy arrays
            a = np.array(embedding1)
            b = np.array(embedding2)
            
            # Compute cosine similarity
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to compute similarity: {e}")
            return 0.0
    
    async def find_similar_texts(
        self,
        query_text: str,
        candidate_texts: List[str],
        top_k: int = 5,
        threshold: float = 0.7,
    ) -> List[Dict[str, Any]]:
        """Find similar texts to a query."""
        try:
            # Generate embeddings
            query_embedding = await self.embed_text(query_text)
            candidate_embeddings = await self.embed_texts(candidate_texts)
            
            # Compute similarities
            similarities = []
            for i, candidate_embedding in enumerate(candidate_embeddings):
                similarity = await self.compute_similarity(query_embedding, candidate_embedding)
                if similarity >= threshold:
                    similarities.append({
                        'text': candidate_texts[i],
                        'similarity': similarity,
                        'index': i,
                    })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            logger.error(f"Failed to find similar texts: {e}")
            return [] 
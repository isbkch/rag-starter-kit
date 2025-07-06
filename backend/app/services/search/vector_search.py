"""
Vector search engine for semantic similarity search.
"""

import logging
from typing import List, Dict, Any, Optional
import time

from app.models.search import SearchResult, SearchResponse, SearchType
from app.services.vectordb.factory import VectorDBFactory
from app.services.vectordb.base import VectorSearchResult
from app.services.search.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class VectorSearchEngine:
    """Vector-based semantic search engine."""
    
    def __init__(self, vector_db=None, embedding_service=None, collection_name=None):
        self.vector_db = vector_db
        self.embedding_service = embedding_service
        self.collection_name = collection_name
        self._connected = False
    
    async def initialize(self):
        """Initialize the search engine."""
        if self.vector_db is None:
            raise ValueError("Vector database not provided to VectorSearchEngine")
        if self.embedding_service is None:
            raise ValueError("Embedding service not provided to VectorSearchEngine")
        
        if not self._connected:
            await self.vector_db.connect()
            self._connected = True
    
    async def search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> SearchResponse:
        """
        Perform vector similarity search.
        
        Args:
            query: Search query text
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            collection_name: Optional collection name
            
        Returns:
            List of search results
        """
        start_time = time.time()
        metrics = get_metrics_collector()
        
        try:
            await self.initialize()
            
            # Generate query embedding
            query_embedding = await self.embedding_service.get_embedding(query)
            
            # Perform vector search
            vector_results = await self.vector_db.search_vectors(
                query_vector=query_embedding,
                limit=limit * 2,  # Get more results to filter by threshold
                filters=filters,
                collection_name=collection_name,
            )
            
            # Convert to SearchResult format and filter by threshold
            search_results = []
            for i, result in enumerate(vector_results):
                if result.score >= min_score:
                    search_result = SearchResult(
                        content=result.content,
                        score=result.score,
                        metadata=result.metadata,
                        source=result.metadata.get('source', ''),
                        title=result.metadata.get('title', ''),
                        context=result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        citations=self._extract_citations_list(result)
                    )
                    search_results.append(search_result)
                
                # Stop if we have enough results
                if len(search_results) >= limit:
                    break
            
            search_time = time.time() - start_time
            logger.info(f"Vector search completed in {search_time:.2f}s, found {len(search_results)} results")
            
            # Record successful vector search metrics
            provider = getattr(self.vector_db, '__class__', {}).get('__name__', 'unknown')
            metrics.record_vector_operation("search", provider, search_time, success=True)
            
            return SearchResponse(
                query=query,
                results=search_results,
                total_results=len(search_results),
                search_time=search_time,
                search_type="vector"
            )
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            
            # Record failed vector search metrics
            search_time = time.time() - start_time
            provider = getattr(self.vector_db, '__class__', {}).get('__name__', 'unknown')
            metrics.record_vector_operation("search", provider, search_time, success=False)
            metrics.record_error(type(e).__name__, "vector.search")
            
            raise
    
    async def search_by_embedding(
        self,
        query_embedding: List[float],
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None,
        collection_name: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        Perform vector search using pre-computed embedding.
        
        Args:
            query_embedding: Pre-computed query embedding
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            filters: Optional metadata filters
            collection_name: Optional collection name
            
        Returns:
            List of search results
        """
        start_time = time.time()
        
        try:
            await self.initialize()
            
            # Perform vector search
            vector_results = await self.vector_db.search_vectors(
                query_vector=query_embedding,
                limit=limit * 2,
                filters=filters,
                collection_name=collection_name,
            )
            
            # Convert and filter results
            search_results = []
            for result in vector_results:
                if result.score >= similarity_threshold:
                    search_result = SearchResult(
                        id=result.id,
                        document_id=result.metadata.get('document_id', ''),
                        chunk_id=result.metadata.get('chunk_id', ''),
                        content=result.content,
                        score=result.score,
                        similarity_score=result.score,
                        keyword_score=None,
                        metadata=result.metadata,
                        highlights=[],
                        citations=self._extract_citations(result),
                    )
                    search_results.append(search_result)
                
                if len(search_results) >= limit:
                    break
            
            search_time = (time.time() - start_time) * 1000
            logger.debug(f"Vector search by embedding completed in {search_time:.2f}ms")
            
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search by embedding failed: {e}")
            raise
    
    async def find_similar_chunks(
        self,
        chunk_id: str,
        limit: int = 5,
        similarity_threshold: float = 0.8,
        exclude_same_document: bool = True,
    ) -> List[SearchResult]:
        """
        Find chunks similar to a given chunk.
        
        Args:
            chunk_id: ID of the reference chunk
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            exclude_same_document: Whether to exclude chunks from same document
            
        Returns:
            List of similar chunks
        """
        try:
            await self.initialize()
            
            # Get the reference chunk
            reference_chunk = await self.vector_db.get_vector(chunk_id)
            if not reference_chunk or not reference_chunk.embedding:
                raise ValueError(f"Chunk {chunk_id} not found or has no embedding")
            
            # Prepare filters
            filters = {}
            if exclude_same_document:
                document_id = reference_chunk.metadata.get('document_id')
                if document_id:
                    filters['document_id'] = {'$ne': document_id}
            
            # Search for similar chunks
            return await self.search_by_embedding(
                query_embedding=reference_chunk.embedding,
                limit=limit,
                similarity_threshold=similarity_threshold,
                filters=filters,
            )
            
        except Exception as e:
            logger.error(f"Failed to find similar chunks for {chunk_id}: {e}")
            raise
    
    async def get_document_chunks(
        self,
        document_id: str,
        limit: int = 100,
    ) -> List[SearchResult]:
        """
        Get all chunks for a specific document.
        
        Args:
            document_id: Document ID
            limit: Maximum number of chunks
            
        Returns:
            List of document chunks
        """
        try:
            await self.initialize()
            
            # Use zero vector for search with document filter
            zero_vector = [0.0] * self.embedding_service.get_embedding_dimension()
            
            vector_results = await self.vector_db.search_vectors(
                query_vector=zero_vector,
                limit=limit,
                filters={'document_id': document_id},
            )
            
            # Convert to SearchResult format
            search_results = []
            for result in vector_results:
                search_result = SearchResult(
                    id=result.id,
                    document_id=result.metadata.get('document_id', ''),
                    chunk_id=result.metadata.get('chunk_id', ''),
                    content=result.content,
                    score=1.0,  # Not based on similarity
                    similarity_score=None,
                    keyword_score=None,
                    metadata=result.metadata,
                    highlights=[],
                    citations=self._extract_citations(result),
                )
                search_results.append(search_result)
            
            # Sort by chunk index
            search_results.sort(key=lambda x: x.metadata.get('chunk_index', 0))
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to get chunks for document {document_id}: {e}")
            raise
    
    def _extract_citations(self, result: VectorSearchResult) -> List[Dict[str, Any]]:
        """Extract citation information from search result."""
        citations = []
        
        metadata = result.metadata
        if metadata:
            citation = {
                'document_id': metadata.get('document_id', ''),
                'chunk_id': metadata.get('chunk_id', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'start_char': metadata.get('start_char', 0),
                'end_char': metadata.get('end_char', 0),
                'score': result.score,
            }
            
            # Add document metadata if available
            if 'title' in metadata:
                citation['title'] = metadata['title']
            if 'author' in metadata:
                citation['author'] = metadata['author']
            if 'created_at' in metadata:
                citation['created_at'] = metadata['created_at']
            
            citations.append(citation)
        
        return citations
    
    def _extract_citations_list(self, result: VectorSearchResult) -> List[str]:
        """Extract citations as a list of strings."""
        citations = []
        
        metadata = result.metadata
        if metadata:
            # Add source as citation
            source = metadata.get('source', '')
            if source:
                citations.append(source)
            
            # Add title as citation if different from source
            title = metadata.get('title', '')
            if title and title != source:
                citations.append(title)
        
        return citations
    
    async def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents in the vector database.
        
        Args:
            documents: List of documents to index, each containing:
                - content: Text content
                - metadata: Document metadata including id, title, source, etc.
        """
        try:
            await self.initialize()
            
            logger.info(f"Indexing {len(documents)} documents in vector database...")
            
            # Process documents for vector storage
            processed_docs = []
            for doc in documents:
                content = doc.get('content', '')
                if not content:
                    continue
                
                # Generate embedding for the content
                embedding = await self.embedding_service.get_embedding(content)
                
                # Prepare document for vector storage
                vector_doc = {
                    'id': doc.get('id', ''),
                    'content': content,
                    'embedding': embedding,
                    'metadata': doc.get('metadata', {})
                }
                
                # Ensure essential metadata fields
                if 'document_id' not in vector_doc['metadata']:
                    vector_doc['metadata']['document_id'] = doc.get('document_id', doc.get('id', ''))
                if 'chunk_id' not in vector_doc['metadata']:
                    vector_doc['metadata']['chunk_id'] = doc.get('chunk_id', doc.get('id', ''))
                if 'source' not in vector_doc['metadata']:
                    vector_doc['metadata']['source'] = doc.get('source', '')
                if 'title' not in vector_doc['metadata']:
                    vector_doc['metadata']['title'] = doc.get('title', '')
                
                processed_docs.append(vector_doc)
            
            # Store in vector database
            if processed_docs:
                await self.vector_db.add_vectors(
                    vectors=processed_docs,
                    collection_name=self.collection_name
                )
                logger.info(f"Successfully indexed {len(processed_docs)} documents in vector database")
            else:
                logger.warning("No valid documents to index in vector database")
                
        except Exception as e:
            logger.error(f"Error indexing documents in vector database: {e}")
            raise
    
    async def get_search_suggestions(
        self,
        query: str,
        limit: int = 5,
    ) -> List[str]:
        """
        Get search suggestions based on similar content.
        
        Args:
            query: Partial query
            limit: Maximum number of suggestions
            
        Returns:
            List of suggested queries
        """
        try:
            if len(query.strip()) < 3:
                return []
            
            # Perform a search with low threshold to get diverse results
            results = await self.search(
                query=query,
                limit=limit * 2,
                min_score=0.3,
            )
            
            # Extract key phrases from results
            suggestions = set()
            for result in results:
                content = result.content.lower()
                
                # Simple extraction of phrases containing query terms
                words = query.lower().split()
                for word in words:
                    if word in content:
                        # Find sentences containing the word
                        sentences = content.split('.')
                        for sentence in sentences:
                            if word in sentence and len(sentence.strip()) > 10:
                                # Extract key phrases (simplified)
                                phrases = sentence.strip().split(',')
                                for phrase in phrases:
                                    if word in phrase and len(phrase.strip()) < 50:
                                        suggestions.add(phrase.strip())
                
                if len(suggestions) >= limit:
                    break
            
            return list(suggestions)[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get search suggestions: {e}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector search engine statistics."""
        try:
            await self.initialize()
            
            collection_stats = await self.vector_db.get_collection_stats()
            embedding_stats = await self.embedding_service.get_stats()
            
            return {
                'vector_db': collection_stats,
                'embedding_service': embedding_stats,
                'search_type': SearchType.VECTOR.value,
            }
            
        except Exception as e:
            logger.error(f"Failed to get vector search stats: {e}")
            return {}
    
    async def cleanup(self):
        """Clean up resources."""
        if self._connected:
            await self.vector_db.disconnect()
            self._connected = False

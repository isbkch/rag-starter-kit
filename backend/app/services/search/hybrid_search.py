"""
Hybrid search engine combining vector search and keyword search.
"""
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
import numpy as np

from app.models.search import SearchResult, SearchResponse
from app.services.search.vector_search import VectorSearchEngine
from app.services.search.keyword_search import KeywordSearchEngine

logger = logging.getLogger(__name__)


class HybridSearchEngine:
    """Hybrid search engine combining vector and keyword search."""

    def __init__(
        self,
        vector_search_engine: VectorSearchEngine,
        keyword_search_engine: KeywordSearchEngine,
        vector_weight: float = 0.7,
        keyword_weight: float = 0.3,
        fusion_method: str = "rrf",  # "rrf" (Reciprocal Rank Fusion) or "weighted"
    ):
        self.vector_search = vector_search_engine
        self.keyword_search = keyword_search_engine
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self.fusion_method = fusion_method
        self.executor = ThreadPoolExecutor(max_workers=2)

        # Validate weights
        if not (0 <= vector_weight <= 1 and 0 <= keyword_weight <= 1):
            raise ValueError("Search weights must be between 0 and 1")

        # Normalize weights
        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight

    async def search(
        self,
        query: str,
        limit: int = 10,
        vector_limit: int = None,
        keyword_limit: int = None,
        min_vector_score: float = 0.0,
        min_keyword_score: float = 0.0,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SearchResponse:
        """Perform hybrid search combining vector and keyword search."""
        try:
            import time

            start_time = time.time()

            # Set default limits for individual searches
            if vector_limit is None:
                vector_limit = limit * 2
            if keyword_limit is None:
                keyword_limit = limit * 2

            # Perform both searches concurrently
            vector_task = asyncio.create_task(
                self._perform_vector_search(
                    query, vector_limit, min_vector_score, filters
                )
            )
            keyword_task = asyncio.create_task(
                self._perform_keyword_search(query, keyword_limit, min_keyword_score)
            )

            vector_response, keyword_response = await asyncio.gather(
                vector_task, keyword_task, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(vector_response, Exception):
                logger.error(f"Vector search failed: {vector_response}")
                vector_response = SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    search_type="vector",
                )

            if isinstance(keyword_response, Exception):
                logger.error(f"Keyword search failed: {keyword_response}")
                keyword_response = SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    search_type="keyword",
                )

            # Fuse results
            fused_results = self._fuse_results(
                vector_response.results, keyword_response.results, limit
            )

            search_time = time.time() - start_time

            return SearchResponse(
                query=query,
                results=fused_results,
                total_results=len(fused_results),
                search_time=search_time,
                search_type="hybrid",
                metadata={
                    "vector_results": len(vector_response.results),
                    "keyword_results": len(keyword_response.results),
                    "vector_time": vector_response.search_time,
                    "keyword_time": keyword_response.search_time,
                    "fusion_method": self.fusion_method,
                    "vector_weight": self.vector_weight,
                    "keyword_weight": self.keyword_weight,
                },
            )

        except Exception as e:
            logger.error(f"Error performing hybrid search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0,
                search_type="hybrid",
                error=str(e),
            )

    async def _perform_vector_search(
        self,
        query: str,
        limit: int,
        min_score: float,
        filters: Optional[Dict[str, Any]],
    ) -> SearchResponse:
        """Perform vector search asynchronously."""
        return await self.vector_search.search(
            query=query, limit=limit, min_score=min_score, filters=filters
        )

    async def _perform_keyword_search(
        self, query: str, limit: int, min_score: float
    ) -> SearchResponse:
        """Perform keyword search asynchronously."""
        return await self.keyword_search.search(
            query=query, limit=limit, min_score=min_score
        )

    def _fuse_results(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        limit: int,
    ) -> List[SearchResult]:
        """Fuse vector and keyword search results."""
        if self.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(vector_results, keyword_results, limit)
        elif self.fusion_method == "weighted":
            return self._weighted_score_fusion(vector_results, keyword_results, limit)
        else:
            logger.warning(f"Unknown fusion method: {self.fusion_method}, using RRF")
            return self._reciprocal_rank_fusion(vector_results, keyword_results, limit)

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        limit: int,
        k: int = 60,
    ) -> List[SearchResult]:
        """Fuse results using Reciprocal Rank Fusion (RRF)."""
        # Create a mapping of content to results for deduplication
        result_map = {}

        # Process vector results
        for rank, result in enumerate(vector_results):
            content_key = self._get_content_key(result)
            rrf_score = self.vector_weight / (k + rank + 1)

            if content_key in result_map:
                result_map[content_key]["rrf_score"] += rrf_score
                result_map[content_key]["sources"].add("vector")
            else:
                result_map[content_key] = {
                    "result": result,
                    "rrf_score": rrf_score,
                    "sources": {"vector"},
                    "vector_rank": rank + 1,
                    "keyword_rank": None,
                }

        # Process keyword results
        for rank, result in enumerate(keyword_results):
            content_key = self._get_content_key(result)
            rrf_score = self.keyword_weight / (k + rank + 1)

            if content_key in result_map:
                result_map[content_key]["rrf_score"] += rrf_score
                result_map[content_key]["sources"].add("keyword")
                result_map[content_key]["keyword_rank"] = rank + 1
            else:
                result_map[content_key] = {
                    "result": result,
                    "rrf_score": rrf_score,
                    "sources": {"keyword"},
                    "vector_rank": None,
                    "keyword_rank": rank + 1,
                }

        # Sort by RRF score and create final results
        sorted_items = sorted(
            result_map.values(), key=lambda x: x["rrf_score"], reverse=True
        )

        fused_results = []
        for item in sorted_items[:limit]:
            result = item["result"]
            # Update result with hybrid information
            result.score = item["rrf_score"]
            result.metadata = result.metadata or {}
            result.metadata.update(
                {
                    "hybrid_sources": list(item["sources"]),
                    "vector_rank": item["vector_rank"],
                    "keyword_rank": item["keyword_rank"],
                    "rrf_score": item["rrf_score"],
                }
            )
            fused_results.append(result)

        return fused_results

    def _weighted_score_fusion(
        self,
        vector_results: List[SearchResult],
        keyword_results: List[SearchResult],
        limit: int,
    ) -> List[SearchResult]:
        """Fuse results using weighted score combination."""
        result_map = {}

        # Normalize scores within each result set
        vector_scores = [r.score for r in vector_results] if vector_results else [0]
        keyword_scores = [r.score for r in keyword_results] if keyword_results else [0]

        vector_max = max(vector_scores) if vector_scores else 1
        keyword_max = max(keyword_scores) if keyword_scores else 1

        # Process vector results
        for result in vector_results:
            content_key = self._get_content_key(result)
            normalized_score = result.score / vector_max if vector_max > 0 else 0
            weighted_score = normalized_score * self.vector_weight

            if content_key in result_map:
                result_map[content_key]["weighted_score"] += weighted_score
                result_map[content_key]["sources"].add("vector")
                result_map[content_key]["vector_score"] = normalized_score
            else:
                result_map[content_key] = {
                    "result": result,
                    "weighted_score": weighted_score,
                    "sources": {"vector"},
                    "vector_score": normalized_score,
                    "keyword_score": 0,
                }

        # Process keyword results
        for result in keyword_results:
            content_key = self._get_content_key(result)
            normalized_score = result.score / keyword_max if keyword_max > 0 else 0
            weighted_score = normalized_score * self.keyword_weight

            if content_key in result_map:
                result_map[content_key]["weighted_score"] += weighted_score
                result_map[content_key]["sources"].add("keyword")
                result_map[content_key]["keyword_score"] = normalized_score
            else:
                result_map[content_key] = {
                    "result": result,
                    "weighted_score": weighted_score,
                    "sources": {"keyword"},
                    "vector_score": 0,
                    "keyword_score": normalized_score,
                }

        # Sort by weighted score and create final results
        sorted_items = sorted(
            result_map.values(), key=lambda x: x["weighted_score"], reverse=True
        )

        fused_results = []
        for item in sorted_items[:limit]:
            result = item["result"]
            # Update result with hybrid information
            result.score = item["weighted_score"]
            result.metadata = result.metadata or {}
            result.metadata.update(
                {
                    "hybrid_sources": list(item["sources"]),
                    "vector_score": item["vector_score"],
                    "keyword_score": item["keyword_score"],
                    "weighted_score": item["weighted_score"],
                }
            )
            fused_results.append(result)

        return fused_results

    def _get_content_key(self, result: SearchResult) -> str:
        """Generate a key for deduplication based on content."""
        # Use a combination of content hash and metadata for deduplication
        content_hash = hash(result.content[:200])  # Use first 200 chars
        source = result.metadata.get("source", "") if result.metadata else ""
        chunk_index = result.metadata.get("chunk_index", 0) if result.metadata else 0
        return f"{content_hash}_{source}_{chunk_index}"

    def update_weights(self, vector_weight: float, keyword_weight: float):
        """Update search weights."""
        if not (0 <= vector_weight <= 1 and 0 <= keyword_weight <= 1):
            raise ValueError("Search weights must be between 0 and 1")

        total_weight = vector_weight + keyword_weight
        if total_weight > 0:
            self.vector_weight = vector_weight / total_weight
            self.keyword_weight = keyword_weight / total_weight
        else:
            raise ValueError("At least one weight must be greater than 0")

    def get_stats(self) -> Dict[str, Any]:
        """Get hybrid search engine statistics."""
        return {
            "vector_stats": self.vector_search.get_stats(),
            "keyword_stats": self.keyword_search.get_stats(),
            "vector_weight": self.vector_weight,
            "keyword_weight": self.keyword_weight,
            "fusion_method": self.fusion_method,
            "engine_type": "hybrid",
        }

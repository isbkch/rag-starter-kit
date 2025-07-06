"""
Search endpoints with streaming responses.
"""
import logging
import json
from typing import Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.models.search import SearchResponse, SearchRequest, SearchType
from app.services.search.search_manager import get_search_manager, SearchManager
from app.core.config import get_settings, Settings
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()


# Extend SearchRequest for streaming support
class StreamingSearchRequest(SearchRequest):
    """Extended search request with streaming support."""

    stream: bool = Field(False, description="Enable streaming response")


class SearchStatsRequest(BaseModel):
    """Search statistics request model."""

    include_detailed: bool = Field(False, description="Include detailed statistics")


async def get_search_manager_dep(
    settings: Settings = Depends(get_settings),
) -> SearchManager:
    """Dependency to get search manager."""
    return await get_search_manager(settings)


@router.post("/", response_model=SearchResponse)
async def search(
    request: StreamingSearchRequest,
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Perform search with optional streaming."""
    try:
        if request.stream:
            return StreamingResponse(
                stream_search_results(search_manager, request),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-cache"},
            )
        else:
            # Regular search
            response = await search_manager.search(
                query=request.query,
                search_type=request.search_type,
                limit=request.max_results,
                filters=request.filters,
                min_score=request.similarity_threshold,
            )
            return response

    except Exception as e:
        logger.error(f"Error performing search: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {str(e)}")


async def stream_search_results(
    search_manager: SearchManager, request: StreamingSearchRequest
):
    """Stream search results as NDJSON."""
    try:
        # Send initial metadata
        yield json.dumps(
            {
                "type": "metadata",
                "query": request.query,
                "search_type": request.search_type,
                "limit": request.max_results,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
        ) + "\n"

        # Perform search
        response = await search_manager.search(
            query=request.query,
            search_type=request.search_type,
            limit=request.max_results,
            filters=request.filters,
            min_score=request.similarity_threshold,
        )

        # Stream each result
        for i, result in enumerate(response.results):
            yield json.dumps(
                {
                    "type": "result",
                    "index": i,
                    "data": {
                        "content": result.content,
                        "score": result.score,
                        "metadata": result.metadata,
                        "source": result.source,
                        "title": result.title,
                        "context": result.context,
                        "citations": result.citations,
                    },
                }
            ) + "\n"

        # Send final summary
        yield json.dumps(
            {
                "type": "summary",
                "total_results": response.total_results,
                "search_time": response.search_time,
                "search_type": response.search_type,
                "metadata": response.metadata,
            }
        ) + "\n"

    except Exception as e:
        logger.error(f"Error streaming search results: {e}")
        yield json.dumps({"type": "error", "error": str(e)}) + "\n"


@router.get("/suggestions")
async def get_search_suggestions(
    q: str = Query(..., description="Partial query for suggestions"),
    limit: int = Query(5, ge=1, le=20, description="Maximum number of suggestions"),
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Get search suggestions based on partial query."""
    try:
        # In a production system, you'd implement proper suggestion logic
        # For now, return basic suggestions
        suggestions = [
            f"{q} documentation",
            f"{q} examples",
            f"{q} tutorial",
            f"{q} best practices",
            f"{q} troubleshooting",
        ][:limit]

        return {"query": q, "suggestions": suggestions, "total": len(suggestions)}

    except Exception as e:
        logger.error(f"Error getting search suggestions: {e}")
        raise HTTPException(status_code=500, detail=f"Suggestions error: {str(e)}")


@router.post("/stats")
async def get_search_stats(
    request: SearchStatsRequest,
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Get search engine statistics."""
    try:
        stats = search_manager.get_search_stats()

        if not request.include_detailed:
            # Return simplified stats
            simplified_stats = {
                "initialized": stats.get("initialized", False),
                "vector_db_provider": stats.get("vector_db_provider"),
                "embedding_provider": stats.get("embedding_provider"),
            }

            if "vector_search" in stats:
                simplified_stats["total_documents"] = stats["vector_search"].get(
                    "indexed_documents", 0
                )

            if "keyword_search" in stats:
                simplified_stats["keyword_index_size"] = stats["keyword_search"].get(
                    "indexed_documents", 0
                )

            return simplified_stats

        return stats

    except Exception as e:
        logger.error(f"Error getting search stats: {e}")
        raise HTTPException(status_code=500, detail=f"Stats error: {str(e)}")


@router.post("/health")
async def search_health_check(
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Perform health check on search components."""
    try:
        health = await search_manager.health_check()

        # Set appropriate HTTP status based on health
        status_code = 200
        if health.get("status") == "degraded":
            status_code = 206  # Partial Content
        elif health.get("status") == "unhealthy":
            status_code = 503  # Service Unavailable

        return health

    except Exception as e:
        logger.error(f"Error performing search health check: {e}")
        return {"status": "unhealthy", "error": str(e)}


@router.post("/reindex")
async def reindex_all_documents(
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Trigger reindexing of all documents."""
    try:
        # In a production system, you'd:
        # 1. Queue a background job for reindexing
        # 2. Return a job ID for status tracking
        # 3. Implement proper job management

        logger.info("Full reindexing requested")

        return {
            "message": "Full reindexing initiated",
            "status": "started",
            "job_id": "reindex_"
            + str(hash(str(id(search_manager)))),  # Placeholder job ID
        }

    except Exception as e:
        logger.error(f"Error initiating reindex: {e}")
        raise HTTPException(status_code=500, detail=f"Reindex error: {str(e)}")


@router.put("/weights")
async def update_hybrid_weights(
    vector_weight: float = Query(
        ..., ge=0.0, le=1.0, description="Vector search weight"
    ),
    keyword_weight: float = Query(
        ..., ge=0.0, le=1.0, description="Keyword search weight"
    ),
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Update hybrid search weights."""
    try:
        search_manager.update_hybrid_weights(vector_weight, keyword_weight)

        return {
            "message": "Hybrid search weights updated",
            "vector_weight": vector_weight,
            "keyword_weight": keyword_weight,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating hybrid weights: {e}")
        raise HTTPException(status_code=500, detail=f"Weight update error: {str(e)}")


@router.post("/vector")
async def vector_search(
    request: StreamingSearchRequest,
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Perform vector-only search."""
    try:
        request.search_type = SearchType.VECTOR
        return await search(request, search_manager)
    except Exception as e:
        logger.error(f"Error performing vector search: {e}")
        raise HTTPException(status_code=500, detail=f"Vector search error: {str(e)}")


@router.post("/keyword")
async def keyword_search(
    request: StreamingSearchRequest,
    search_manager: SearchManager = Depends(get_search_manager_dep),
):
    """Perform keyword-only search."""
    try:
        request.search_type = SearchType.KEYWORD
        return await search(request, search_manager)
    except Exception as e:
        logger.error(f"Error performing keyword search: {e}")
        raise HTTPException(status_code=500, detail=f"Keyword search error: {str(e)}")

"""
Search endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.post("/")
async def search():
    """Perform hybrid search."""
    # TODO: Implement search functionality
    return {"results": [], "total": 0}


@router.post("/vector")
async def vector_search():
    """Perform vector-only search."""
    # TODO: Implement vector search
    return {"results": [], "total": 0} 
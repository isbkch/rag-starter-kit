"""
Document management endpoints.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def list_documents():
    """List all documents."""
    # TODO: Implement document listing
    return {"documents": [], "total": 0}


@router.post("/upload")
async def upload_document():
    """Upload a new document."""
    # TODO: Implement document upload
    return {"message": "Document upload endpoint - to be implemented"} 
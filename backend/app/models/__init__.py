"""
Pydantic models for the RAG platform.
"""

from .documents import (
    Document,
    DocumentChunk,
    DocumentListResponse,
    DocumentMetadata,
    DocumentProcessingRequest,
    DocumentProcessingResponse,
    DocumentStatus,
    DocumentType,
    DocumentUploadRequest,
    DocumentUploadResponse,
)
from .search import (
    AdvancedSearchRequest,
    SearchAggregation,
    SearchAnalytics,
    SearchFacet,
    SearchFilter,
    SearchRequest,
    SearchResponse,
    SearchResult,
    SearchType,
)

__all__ = [
    # Documents
    "Document",
    "DocumentChunk",
    "DocumentMetadata",
    "DocumentType",
    "DocumentStatus",
    "DocumentUploadRequest",
    "DocumentUploadResponse",
    "DocumentProcessingRequest",
    "DocumentProcessingResponse",
    "DocumentListResponse",
    # Search
    "SearchRequest",
    "SearchResponse",
    "SearchResult",
    "SearchType",
    "SearchFilter",
    "SearchFacet",
    "SearchAggregation",
    "AdvancedSearchRequest",
    "SearchAnalytics",
]

"""
Search-related Pydantic models.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class SearchType(str, Enum):
    """Search types."""
    VECTOR = "vector"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: Optional[Dict[str, Any]] = None
    include_metadata: bool = True
    include_embeddings: bool = False


class SearchResult(BaseModel):
    """Individual search result model."""
    content: str
    score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    source: str = Field(default="")
    title: str = Field(default="")
    context: str = Field(default="")
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    similarity_score: Optional[float] = None
    keyword_score: Optional[float] = None
    highlights: List[str] = Field(default_factory=list)
    
    # Legacy fields for backward compatibility
    id: Optional[str] = None
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    search_type: str
    results: List[SearchResult]
    total_results: int
    search_time: float
    metadata: Dict[str, Any] = Field(default_factory=dict)
    filters_applied: Optional[Dict[str, Any]] = None
    suggestions: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    error: Optional[str] = None
    
    # Legacy field for backward compatibility
    search_time_ms: Optional[float] = None
    
    def __init__(self, **data):
        # Handle backward compatibility
        if 'search_time_ms' in data and 'search_time' not in data:
            data['search_time'] = data['search_time_ms'] / 1000.0
        super().__init__(**data)


class SearchFilter(BaseModel):
    """Search filter model."""
    field: str
    operator: str  # eq, ne, gt, lt, gte, lte, in, nin, contains
    value: Any


class SearchFacet(BaseModel):
    """Search facet model."""
    field: str
    values: List[Dict[str, Any]]  # [{"value": "pdf", "count": 10}]


class SearchAggregation(BaseModel):
    """Search aggregation model."""
    name: str
    type: str  # terms, date_histogram, range
    field: str
    results: List[Dict[str, Any]]


class AdvancedSearchRequest(BaseModel):
    """Advanced search request model."""
    query: str
    search_type: SearchType = SearchType.HYBRID
    max_results: int = Field(default=10, ge=1, le=100)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: List[SearchFilter] = Field(default_factory=list)
    facets: List[str] = Field(default_factory=list)
    aggregations: List[SearchAggregation] = Field(default_factory=list)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="desc", regex="^(asc|desc)$")
    include_metadata: bool = True
    include_embeddings: bool = False
    highlight_fields: List[str] = Field(default_factory=list)


class SearchAnalytics(BaseModel):
    """Search analytics model."""
    query: str
    search_type: SearchType
    results_count: int
    search_time_ms: float
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    clicked_results: List[str] = Field(default_factory=list)
    user_feedback: Optional[str] = None 
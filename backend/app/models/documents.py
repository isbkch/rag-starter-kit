"""
Document-related Pydantic models.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TXT = "txt"


class DocumentStatus(str, Enum):
    """Document processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentMetadata(BaseModel):
    """Document metadata model."""
    title: Optional[str] = None
    author: Optional[str] = None
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    file_size: Optional[int] = None
    page_count: Optional[int] = None
    language: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    custom_fields: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """Document chunk model."""
    id: str
    document_id: str
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Document(BaseModel):
    """Document model."""
    id: str
    filename: str
    original_filename: str
    file_path: str
    document_type: DocumentType
    status: DocumentStatus
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = Field(default_factory=list)
    total_chunks: int = 0
    processed_chunks: int = 0
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    filename: str
    file_size: int
    document_type: DocumentType
    metadata: Optional[DocumentMetadata] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    document_id: str
    upload_url: Optional[str] = None
    status: DocumentStatus
    message: str


class DocumentProcessingRequest(BaseModel):
    """Document processing request model."""
    document_id: str
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None
    extract_metadata: bool = True


class DocumentProcessingResponse(BaseModel):
    """Document processing response model."""
    document_id: str
    status: DocumentStatus
    total_chunks: int
    processed_chunks: int
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[Document]
    total: int
    page: int
    page_size: int
    total_pages: int 
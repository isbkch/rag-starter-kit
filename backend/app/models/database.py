"""
SQLAlchemy database models for the RAG platform.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, JSON, Float, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class Document(Base):
    """Document model for storing document metadata."""
    
    __tablename__ = "documents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(255), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=True)
    file_size = Column(Integer, nullable=False)
    file_hash = Column(String(64), nullable=True, index=True)
    document_type = Column(String(50), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)
    
    # Processing metadata
    total_chunks = Column(Integer, default=0)
    processed_chunks = Column(Integer, default=0)
    processing_time = Column(Float, nullable=True)
    
    # Content metadata
    metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    processed_at = Column(DateTime, nullable=True)
    
    # Relationships
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Document(id={self.id}, title='{self.title}', status='{self.status}')>"


class DocumentChunk(Base):
    """Document chunk model for storing processed text chunks."""
    
    __tablename__ = "document_chunks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    content_hash = Column(String(64), nullable=True, index=True)
    
    # Chunk metadata
    chunk_size = Column(Integer, nullable=False)
    start_char = Column(Integer, nullable=True)
    end_char = Column(Integer, nullable=True)
    metadata = Column(JSON, nullable=True)
    
    # Vector database references
    vector_id = Column(String(255), nullable=True, index=True)
    embedding_model = Column(String(100), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")
    
    def __repr__(self):
        return f"<DocumentChunk(id={self.id}, document_id={self.document_id}, chunk_index={self.chunk_index})>"


class SearchQuery(Base):
    """Search query model for storing search history and analytics."""
    
    __tablename__ = "search_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_text = Column(Text, nullable=False)
    search_type = Column(String(50), nullable=False)
    
    # Search parameters
    limit = Column(Integer, nullable=False, default=10)
    min_score = Column(Float, nullable=True)
    filters = Column(JSON, nullable=True)
    
    # Results metadata
    total_results = Column(Integer, nullable=False, default=0)
    search_time = Column(Float, nullable=False)
    
    # User context
    user_id = Column(String(255), nullable=True, index=True)
    session_id = Column(String(255), nullable=True, index=True)
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<SearchQuery(id={self.id}, query='{self.query_text[:50]}...', type='{self.search_type}')>"


class SearchResult(Base):
    """Search result model for storing individual search results."""
    
    __tablename__ = "search_results"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    query_id = Column(UUID(as_uuid=True), ForeignKey("search_queries.id"), nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    chunk_id = Column(UUID(as_uuid=True), ForeignKey("document_chunks.id"), nullable=True)
    
    # Result metadata
    score = Column(Float, nullable=False)
    rank = Column(Integer, nullable=False)
    result_type = Column(String(50), nullable=False)  # vector, keyword, hybrid
    
    # Content
    content = Column(Text, nullable=False)
    title = Column(String(255), nullable=False)
    source = Column(String(500), nullable=False)
    context = Column(Text, nullable=True)
    
    # Additional metadata
    metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    query = relationship("SearchQuery")
    document = relationship("Document")
    chunk = relationship("DocumentChunk")
    
    def __repr__(self):
        return f"<SearchResult(id={self.id}, score={self.score}, rank={self.rank})>"


class EmbeddingCache(Base):
    """Embedding cache model for storing computed embeddings."""
    
    __tablename__ = "embedding_cache"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    content_hash = Column(String(64), nullable=False, unique=True, index=True)
    content = Column(Text, nullable=False)
    
    # Embedding metadata
    embedding_model = Column(String(100), nullable=False)
    embedding_dimensions = Column(Integer, nullable=False)
    embedding_provider = Column(String(50), nullable=False)
    
    # Cache metadata
    hit_count = Column(Integer, default=0)
    last_accessed = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<EmbeddingCache(id={self.id}, model='{self.embedding_model}', hits={self.hit_count})>"


class SystemMetric(Base):
    """System metrics model for storing performance and health metrics."""
    
    __tablename__ = "system_metrics"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String(100), nullable=False, index=True)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    value = Column(Float, nullable=False)
    
    # Metric metadata
    labels = Column(JSON, nullable=True)
    unit = Column(String(20), nullable=True)
    description = Column(String(255), nullable=True)
    
    # Context
    component = Column(String(50), nullable=True, index=True)
    environment = Column(String(50), nullable=True)
    
    # Timestamps
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    def __repr__(self):
        return f"<SystemMetric(name='{self.metric_name}', value={self.value}, timestamp={self.timestamp})>"


class JobStatus(Base):
    """Job status model for tracking background job execution."""
    
    __tablename__ = "job_status"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(String(255), nullable=False, unique=True, index=True)
    job_type = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False, default="pending")
    
    # Job details
    task_name = Column(String(100), nullable=False)
    args = Column(JSON, nullable=True)
    kwargs = Column(JSON, nullable=True)
    
    # Progress tracking
    progress = Column(Float, default=0.0)
    total_items = Column(Integer, nullable=True)
    processed_items = Column(Integer, default=0)
    
    # Results and errors
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    traceback = Column(Text, nullable=True)
    
    # Timing
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration = Column(Float, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f"<JobStatus(id={self.id}, job_id='{self.job_id}', status='{self.status}')>"

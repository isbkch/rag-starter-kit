"""
Document service for database operations.
"""

from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime
import logging
import uuid

from app.models.database import Document, DocumentChunk
from app.models.documents import DocumentStatus, DocumentType

logger = logging.getLogger(__name__)


class DocumentService:
    """Service for document database operations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def create_document(
        self,
        title: str,
        filename: str,
        original_filename: str,
        file_path: Optional[str] = None,
        file_size: int = 0,
        file_hash: Optional[str] = None,
        document_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> Document:
        """Create a new document record."""
        document = Document(
            id=uuid.uuid4(),
            title=title,
            filename=filename,
            original_filename=original_filename,
            file_path=file_path,
            file_size=file_size,
            file_hash=file_hash,
            document_type=document_type,
            status="pending",
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Created document: {document.id}")
        return document
    
    def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID."""
        return self.db.query(Document).filter(Document.id == document_id).first()
    
    def get_documents(
        self,
        offset: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> tuple[List[Document], int]:
        """Get documents with pagination."""
        query = self.db.query(Document)
        
        if status:
            query = query.filter(Document.status == status)
        
        total = query.count()
        documents = query.order_by(desc(Document.created_at)).offset(offset).limit(limit).all()
        
        return documents, total
    
    def update_document_status(
        self,
        document_id: str,
        status: str,
        error_message: Optional[str] = None,
        processed_at: Optional[datetime] = None,
        processing_time: Optional[float] = None
    ) -> Optional[Document]:
        """Update document status."""
        document = self.get_document(document_id)
        if not document:
            return None
        
        document.status = status
        document.updated_at = datetime.utcnow()
        
        if error_message:
            document.error_message = error_message
        
        if processed_at:
            document.processed_at = processed_at
        
        if processing_time:
            document.processing_time = processing_time
        
        self.db.commit()
        self.db.refresh(document)
        
        logger.info(f"Updated document {document_id} status to {status}")
        return document
    
    def delete_document(self, document_id: str) -> bool:
        """Delete document and all its chunks."""
        document = self.get_document(document_id)
        if not document:
            return False
        
        # Delete chunks first (cascade should handle this, but being explicit)
        self.db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
        
        # Delete document
        self.db.delete(document)
        self.db.commit()
        
        logger.info(f"Deleted document: {document_id}")
        return True
    
    def create_document_chunk(
        self,
        document_id: str,
        chunk_index: int,
        content: str,
        content_hash: Optional[str] = None,
        chunk_size: int = 0,
        start_char: Optional[int] = None,
        end_char: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        vector_id: Optional[str] = None,
        embedding_model: Optional[str] = None
    ) -> DocumentChunk:
        """Create a document chunk."""
        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=document_id,
            chunk_index=chunk_index,
            content=content,
            content_hash=content_hash,
            chunk_size=chunk_size or len(content),
            start_char=start_char,
            end_char=end_char,
            metadata=metadata or {},
            vector_id=vector_id,
            embedding_model=embedding_model,
            created_at=datetime.utcnow()
        )
        
        self.db.add(chunk)
        self.db.commit()
        self.db.refresh(chunk)
        
        return chunk
    
    def get_document_chunks(self, document_id: str) -> List[DocumentChunk]:
        """Get all chunks for a document."""
        return self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).order_by(DocumentChunk.chunk_index).all()
    
    def update_document_chunk_counts(self, document_id: str) -> Optional[Document]:
        """Update document chunk counts."""
        document = self.get_document(document_id)
        if not document:
            return None
        
        chunk_count = self.db.query(DocumentChunk).filter(
            DocumentChunk.document_id == document_id
        ).count()
        
        document.total_chunks = chunk_count
        document.processed_chunks = chunk_count
        document.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(document)
        
        return document
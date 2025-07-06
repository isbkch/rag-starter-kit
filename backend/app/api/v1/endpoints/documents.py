"""
Document management endpoints.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
import os
import tempfile
import uuid
import hashlib
from datetime import datetime

from app.models.documents import DocumentResponse, DocumentUploadResponse, DocumentListResponse
from app.services.ingestion.document_processor import DocumentProcessor
from app.services.search.search_manager import get_search_manager, SearchManager
from app.services.document_service import DocumentService
from app.core.config import get_settings, Settings
from app.core.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

# Supported file types
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.md', '.txt'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB

def get_document_processor() -> DocumentProcessor:
    """Get document processor instance."""
    return DocumentProcessor()

async def get_search_manager_dep(settings: Settings = Depends(get_settings)) -> SearchManager:
    """Dependency to get search manager."""
    return await get_search_manager(settings)

def get_document_service(db: Session = Depends(get_db)) -> DocumentService:
    """Get document service instance."""
    return DocumentService(db)

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    search_manager: SearchManager = Depends(get_search_manager_dep),
    document_service: DocumentService = Depends(get_document_service)
):
    """Upload and process a document."""
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type. Supported types: {', '.join(SUPPORTED_EXTENSIONS)}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Seek back to beginning
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB"
            )
        
        # Read file content and generate hash
        content = await file.read()
        file_hash = hashlib.sha256(content).hexdigest()
        
        # Create document record in database
        document = document_service.create_document(
            title=file.filename,
            filename=file.filename,
            original_filename=file.filename,
            file_size=file_size,
            file_hash=file_hash,
            document_type=file_ext[1:],  # Remove dot from extension
            metadata={"original_size": file_size}
        )
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Update document status to processing
            document_service.update_document_status(str(document.id), "processing")
            
            # Process document
            logger.info(f"Processing document: {file.filename}")
            processed_doc = await document_processor.process_document(
                file_path=temp_file_path,
                original_filename=file.filename,
                generate_embeddings=True,
                store_in_vector_db=True
            )
            
            # Save chunks to database
            for i, chunk in enumerate(processed_doc.chunks):
                document_service.create_document_chunk(
                    document_id=str(document.id),
                    chunk_index=i,
                    content=chunk.content,
                    content_hash=hashlib.sha256(chunk.content.encode()).hexdigest(),
                    chunk_size=len(chunk.content),
                    metadata=chunk.metadata
                )
            
            # Update document with processing results
            document_service.update_document_status(
                str(document.id),
                "completed",
                processed_at=datetime.utcnow(),
                processing_time=0.0  # Would need to track actual processing time
            )
            document_service.update_document_chunk_counts(str(document.id))
            
            # Prepare documents for indexing
            documents_for_indexing = []
            for chunk in processed_doc.chunks:
                documents_for_indexing.append({
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': {
                        **chunk.metadata,
                        'document_id': str(document.id),
                        'filename': file.filename,
                        'upload_time': datetime.utcnow().isoformat()
                    }
                })
            
            # Index documents in background
            background_tasks.add_task(
                search_manager.index_documents,
                documents_for_indexing
            )
            
            return {
                "document_id": str(document.id),
                "filename": file.filename,
                "file_size": file_size,
                "chunks_created": len(processed_doc.chunks),
                "status": "completed",
                "message": "Document uploaded and processed successfully"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        # Update document status to failed if already created
        if 'document' in locals():
            document_service.update_document_status(
                str(document.id), 
                "failed", 
                error_message="HTTP error during processing"
            )
        raise
    except Exception as e:
        # Update document status to failed if already created
        if 'document' in locals():
            document_service.update_document_status(
                str(document.id), 
                "failed", 
                error_message=str(e)
            )
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    status: Optional[str] = None,
    document_service: DocumentService = Depends(get_document_service)
):
    """List all documents."""
    try:
        documents_db, total = document_service.get_documents(
            offset=offset,
            limit=limit,
            status=status
        )
        
        # Convert database documents to response format
        documents = []
        for doc in documents_db:
            documents.append({
                "id": str(doc.id),
                "title": doc.title,
                "filename": doc.filename,
                "file_size": doc.file_size,
                "document_type": doc.document_type,
                "status": doc.status,
                "total_chunks": doc.total_chunks,
                "processed_chunks": doc.processed_chunks,
                "created_at": doc.created_at.isoformat(),
                "updated_at": doc.updated_at.isoformat(),
                "processed_at": doc.processed_at.isoformat() if doc.processed_at else None,
                "error_message": doc.error_message
            })
        
        page = (offset // limit) + 1
        total_pages = (total + limit - 1) // limit if total > 0 else 0
        
        return DocumentListResponse(
            documents=documents,
            total=total,
            page=page,
            page_size=limit,
            total_pages=total_pages
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    document_service: DocumentService = Depends(get_document_service)
):
    """Get document details by ID."""
    try:
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks for content summary
        chunks = document_service.get_document_chunks(document_id)
        content_preview = ""
        if chunks:
            content_preview = chunks[0].content[:500] + "..." if len(chunks[0].content) > 500 else chunks[0].content
        
        return DocumentResponse(
            id=str(document.id),
            title=document.title,
            source=document.filename,
            content=content_preview,
            metadata={
                "file_size": document.file_size,
                "document_type": document.document_type,
                "status": document.status,
                "total_chunks": document.total_chunks,
                "processed_chunks": document.processed_chunks,
                "processing_time": document.processing_time,
                "error_message": document.error_message,
                **document.metadata
            },
            created_at=document.created_at,
            updated_at=document.updated_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    search_manager: SearchManager = Depends(get_search_manager_dep)
):
    """Delete a document."""
    try:
        # Check if document exists
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks to remove from search indices
        chunks = document_service.get_document_chunks(document_id)
        chunk_ids = [str(chunk.id) for chunk in chunks]
        
        # Remove from database (this will cascade to chunks)
        success = document_service.delete_document(document_id)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to delete document from database")
        
        # Remove from search indices in background
        if chunk_ids:
            background_tasks.add_task(
                _remove_from_search_indices,
                search_manager,
                chunk_ids
            )
        
        logger.info(f"Document deleted successfully: {document_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Document {document_id} deleted successfully",
                "document_id": document_id,
                "status": "deleted",
                "chunks_removed": len(chunk_ids)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.post("/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    document_service: DocumentService = Depends(get_document_service),
    search_manager: SearchManager = Depends(get_search_manager_dep)
):
    """Reindex a specific document."""
    try:
        # Check if document exists
        document = document_service.get_document(document_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Get document chunks
        chunks = document_service.get_document_chunks(document_id)
        if not chunks:
            raise HTTPException(status_code=400, detail="Document has no chunks to reindex")
        
        # Prepare documents for reindexing
        documents_for_indexing = []
        for chunk in chunks:
            documents_for_indexing.append({
                'id': str(chunk.id),
                'content': chunk.content,
                'metadata': {
                    **chunk.metadata,
                    'document_id': document_id,
                    'filename': document.filename,
                    'reindex_time': datetime.utcnow().isoformat()
                }
            })
        
        # Reindex documents in background
        background_tasks.add_task(
            search_manager.index_documents,
            documents_for_indexing
        )
        
        logger.info(f"Document reindexing initiated for: {document_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Document {document_id} reindexing initiated",
                "document_id": document_id,
                "status": "reindexing",
                "chunks_to_reindex": len(chunks)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reindexing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reindexing document: {str(e)}")


async def _remove_from_search_indices(search_manager: SearchManager, chunk_ids: List[str]):
    """Background task to remove documents from search indices."""
    try:
        # Remove from vector database
        await search_manager.remove_documents(chunk_ids)
        logger.info(f"Removed {len(chunk_ids)} chunks from search indices")
    except Exception as e:
        logger.error(f"Error removing chunks from search indices: {e}")

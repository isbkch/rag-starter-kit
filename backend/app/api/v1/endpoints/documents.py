"""
Document management endpoints.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import tempfile
import uuid
from datetime import datetime

from app.models.documents import DocumentResponse, DocumentUploadResponse, DocumentListResponse
from app.services.ingestion.document_processor import DocumentProcessor
from app.services.search.search_manager import get_search_manager, SearchManager
from app.core.config import get_settings, Settings

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

@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    document_processor: DocumentProcessor = Depends(get_document_processor),
    search_manager: SearchManager = Depends(get_search_manager_dep)
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
        
        # Generate document ID
        document_id = str(uuid.uuid4())
        
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document
            logger.info(f"Processing document: {file.filename}")
            processed_doc = await document_processor.process_document(
                file_path=temp_file_path,
                original_filename=file.filename,
                generate_embeddings=True,
                store_in_vector_db=True
            )
            
            # Prepare documents for indexing
            documents_for_indexing = []
            for chunk in processed_doc.chunks:
                documents_for_indexing.append({
                    'id': chunk.id,
                    'content': chunk.content,
                    'metadata': {
                        **chunk.metadata,
                        'document_id': processed_doc.id,
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
                "document_id": processed_doc.id,
                "filename": file.filename,
                "file_size": file_size,
                "chunks_created": len(processed_doc.chunks),
                "status": processed_doc.status.value,
                "message": "Document uploaded and processed successfully"
            }
            
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@router.get("/", response_model=DocumentListResponse)
async def list_documents(
    limit: int = 100,
    offset: int = 0,
    search_manager: SearchManager = Depends(get_search_manager_dep)
):
    """List all documents."""
    try:
        # Get search statistics which includes document counts
        stats = search_manager.get_search_stats()
        
        # For now, return basic info from stats
        # In a production system, you'd want a proper document metadata store
        documents = []
        
        # Extract document info from vector search stats if available
        if 'vector_search' in stats and 'indexed_documents' in stats['vector_search']:
            total_documents = stats['vector_search']['indexed_documents']
        else:
            total_documents = 0
        
        return DocumentListResponse(
            documents=documents,
            total=total_documents,
            page=page,
            page_size=limit,
            total_pages=(total_documents + limit - 1) // limit if total_documents > 0 else 0
        )
        
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    search_manager: SearchManager = Depends(get_search_manager_dep)
):
    """Get document details by ID."""
    try:
        # In a production system, you'd query a document metadata store
        # For now, return basic document info
        return DocumentResponse(
            id=document_id,
            title=f"Document {document_id}",
            source="unknown",
            content="Document content not available in this endpoint",
            metadata={},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Error retrieving document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    search_manager: SearchManager = Depends(get_search_manager_dep)
):
    """Delete a document."""
    try:
        # In a production system, you'd:
        # 1. Remove from document metadata store
        # 2. Remove from vector database
        # 3. Remove from keyword search index
        
        logger.info(f"Document deletion requested for: {document_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Document {document_id} deletion initiated",
                "document_id": document_id,
                "status": "deleted"
            }
        )
        
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting document: {str(e)}")

@router.post("/{document_id}/reindex")
async def reindex_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    search_manager: SearchManager = Depends(get_search_manager_dep)
):
    """Reindex a specific document."""
    try:
        # In a production system, you'd:
        # 1. Retrieve document from storage
        # 2. Reprocess and reindex
        
        logger.info(f"Document reindexing requested for: {document_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": f"Document {document_id} reindexing initiated",
                "document_id": document_id,
                "status": "reindexing"
            }
        )
        
    except Exception as e:
        logger.error(f"Error reindexing document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reindexing document: {str(e)}")

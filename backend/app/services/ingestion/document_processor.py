"""
Main document processing service that orchestrates the ingestion pipeline.
"""

import logging
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from app.models.documents import (
    Document,
    DocumentChunk,
    DocumentType,
    DocumentStatus,
    DocumentMetadata,
)
from app.services.ingestion.text_extractor import TextExtractor
from app.services.ingestion.chunk_processor import ChunkProcessor, ChunkingConfig
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing service."""
    
    def __init__(self):
        self.text_extractor = TextExtractor()
        self.chunk_processor = ChunkProcessor()
    
    async def process_document(
        self,
        file_path: str,
        original_filename: str,
        document_type: Optional[DocumentType] = None,
        chunking_strategy: str = 'recursive',
        chunking_config: Optional[ChunkingConfig] = None,
        extract_metadata: bool = True,
    ) -> Document:
        """
        Process a document through the complete ingestion pipeline.
        
        Args:
            file_path: Path to the document file
            original_filename: Original filename
            document_type: Document type (auto-detected if None)
            chunking_strategy: Strategy for text chunking
            chunking_config: Configuration for chunking
            extract_metadata: Whether to extract metadata
            
        Returns:
            Processed Document object
        """
        document_id = str(uuid.uuid4())
        
        try:
            # Auto-detect document type if not provided
            if document_type is None:
                document_type = self.text_extractor.detect_document_type(file_path)
            
            # Create initial document object
            document = Document(
                id=document_id,
                filename=Path(file_path).name,
                original_filename=original_filename,
                file_path=file_path,
                document_type=document_type,
                status=DocumentStatus.PROCESSING,
                metadata=DocumentMetadata(),
                chunks=[],
                total_chunks=0,
                processed_chunks=0,
            )
            
            logger.info(f"Starting processing for document {document_id}: {original_filename}")
            
            # Step 1: Extract text
            text_content = await self._extract_text(document)
            
            # Step 2: Extract metadata if requested
            if extract_metadata:
                document.metadata = await self._extract_metadata(document)
            
            # Step 3: Process chunks
            chunks = await self._process_chunks(
                document_id=document_id,
                text=text_content,
                chunking_strategy=chunking_strategy,
                chunking_config=chunking_config,
            )
            
            # Step 4: Update document with results
            document.chunks = chunks
            document.total_chunks = len(chunks)
            document.processed_chunks = len(chunks)
            document.status = DocumentStatus.COMPLETED
            document.updated_at = datetime.utcnow()
            
            logger.info(f"Successfully processed document {document_id} with {len(chunks)} chunks")
            
            return document
            
        except Exception as e:
            logger.error(f"Failed to process document {document_id}: {e}")
            
            # Update document with error status
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            document.updated_at = datetime.utcnow()
            
            raise
    
    async def _extract_text(self, document: Document) -> str:
        """Extract text from document."""
        try:
            text = self.text_extractor.extract_text(
                document.file_path,
                document.document_type
            )
            
            if not text.strip():
                raise ValueError("No text content extracted from document")
            
            logger.debug(f"Extracted {len(text)} characters from {document.filename}")
            return text
            
        except Exception as e:
            logger.error(f"Text extraction failed for {document.filename}: {e}")
            raise ValueError(f"Failed to extract text: {e}")
    
    async def _extract_metadata(self, document: Document) -> DocumentMetadata:
        """Extract metadata from document."""
        try:
            metadata = self.text_extractor.extract_metadata(
                document.file_path,
                document.document_type
            )
            
            # Add processing metadata
            metadata.custom_fields.update({
                'processing_timestamp': datetime.utcnow().isoformat(),
                'processor_version': '1.0.0',
                'document_id': document.id,
            })
            
            logger.debug(f"Extracted metadata for {document.filename}")
            return metadata
            
        except Exception as e:
            logger.warning(f"Metadata extraction failed for {document.filename}: {e}")
            # Return basic metadata on failure
            return DocumentMetadata(
                file_size=Path(document.file_path).stat().st_size,
                custom_fields={
                    'metadata_extraction_error': str(e),
                    'processing_timestamp': datetime.utcnow().isoformat(),
                }
            )
    
    async def _process_chunks(
        self,
        document_id: str,
        text: str,
        chunking_strategy: str,
        chunking_config: Optional[ChunkingConfig],
    ) -> List[DocumentChunk]:
        """Process text into chunks."""
        try:
            # Optimize chunking config if not provided
            if chunking_config is None:
                chunking_config = self.chunk_processor.optimize_chunking_config(text)
            
            chunks = self.chunk_processor.process_document(
                document_id=document_id,
                text=text,
                chunking_strategy=chunking_strategy,
                config=chunking_config,
            )
            
            # Log chunk statistics
            stats = self.chunk_processor.get_chunk_statistics(chunks)
            logger.info(f"Chunk statistics for {document_id}: {stats}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Chunk processing failed for {document_id}: {e}")
            raise ValueError(f"Failed to process chunks: {e}")
    
    def validate_document(self, file_path: str) -> Dict[str, Any]:
        """
        Validate document before processing.
        
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'file_info': {},
        }
        
        try:
            file_path_obj = Path(file_path)
            
            # Check if file exists
            if not file_path_obj.exists():
                validation_result['is_valid'] = False
                validation_result['errors'].append('File does not exist')
                return validation_result
            
            # Check file size
            file_size = file_path_obj.stat().st_size
            validation_result['file_info']['size'] = file_size
            
            if file_size > settings.MAX_FILE_SIZE:
                validation_result['is_valid'] = False
                validation_result['errors'].append(
                    f'File size ({file_size} bytes) exceeds maximum allowed size ({settings.MAX_FILE_SIZE} bytes)'
                )
            
            if file_size == 0:
                validation_result['is_valid'] = False
                validation_result['errors'].append('File is empty')
            
            # Check file extension
            document_type = self.text_extractor.detect_document_type(file_path)
            validation_result['file_info']['document_type'] = document_type.value
            
            # Check if file is readable
            try:
                with open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            except Exception as e:
                validation_result['is_valid'] = False
                validation_result['errors'].append(f'File is not readable: {e}')
            
            # Document type specific validation
            if document_type == DocumentType.PDF:
                validation_result.update(self._validate_pdf(file_path))
            elif document_type == DocumentType.DOCX:
                validation_result.update(self._validate_docx(file_path))
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f'Validation error: {e}')
        
        return validation_result
    
    def _validate_pdf(self, file_path: str) -> Dict[str, Any]:
        """Validate PDF file."""
        result = {'errors': [], 'warnings': []}
        
        try:
            import PyPDF2
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Check if PDF is encrypted
                if pdf_reader.is_encrypted:
                    result['errors'].append('PDF is encrypted and cannot be processed')
                
                # Check number of pages
                page_count = len(pdf_reader.pages)
                if page_count == 0:
                    result['errors'].append('PDF has no pages')
                elif page_count > 1000:
                    result['warnings'].append(f'PDF has many pages ({page_count}), processing may take time')
                
        except Exception as e:
            result['errors'].append(f'PDF validation error: {e}')
        
        return result
    
    def _validate_docx(self, file_path: str) -> Dict[str, Any]:
        """Validate DOCX file."""
        result = {'errors': [], 'warnings': []}
        
        try:
            from docx import Document as DocxDocument
            doc = DocxDocument(file_path)
            
            # Check if document has content
            if not doc.paragraphs:
                result['warnings'].append('DOCX appears to have no paragraphs')
            
        except Exception as e:
            result['errors'].append(f'DOCX validation error: {e}')
        
        return result
    
    def get_supported_formats(self) -> List[Dict[str, Any]]:
        """Get list of supported document formats."""
        return [
            {
                'type': DocumentType.PDF.value,
                'extensions': ['.pdf'],
                'description': 'Adobe PDF documents',
                'max_size_mb': settings.MAX_FILE_SIZE // (1024 * 1024),
            },
            {
                'type': DocumentType.DOCX.value,
                'extensions': ['.docx'],
                'description': 'Microsoft Word documents',
                'max_size_mb': settings.MAX_FILE_SIZE // (1024 * 1024),
            },
            {
                'type': DocumentType.MARKDOWN.value,
                'extensions': ['.md', '.markdown'],
                'description': 'Markdown documents',
                'max_size_mb': settings.MAX_FILE_SIZE // (1024 * 1024),
            },
            {
                'type': DocumentType.TXT.value,
                'extensions': ['.txt'],
                'description': 'Plain text documents',
                'max_size_mb': settings.MAX_FILE_SIZE // (1024 * 1024),
            },
        ]
    
    def estimate_processing_time(self, file_path: str) -> Dict[str, Any]:
        """Estimate processing time for a document."""
        try:
            file_size = Path(file_path).stat().st_size
            document_type = self.text_extractor.detect_document_type(file_path)
            
            # Rough estimates based on file size and type
            base_time = 0
            if document_type == DocumentType.PDF:
                base_time = file_size / (1024 * 1024) * 5  # 5 seconds per MB for PDF
            elif document_type == DocumentType.DOCX:
                base_time = file_size / (1024 * 1024) * 2  # 2 seconds per MB for DOCX
            else:
                base_time = file_size / (1024 * 1024) * 1  # 1 second per MB for text files
            
            return {
                'estimated_seconds': max(1, int(base_time)),
                'file_size_mb': round(file_size / (1024 * 1024), 2),
                'document_type': document_type.value,
            }
            
        except Exception as e:
            return {
                'estimated_seconds': 30,  # Default estimate
                'error': str(e),
            } 
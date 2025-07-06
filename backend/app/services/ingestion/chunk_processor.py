"""
Text chunking service for document processing.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

import tiktoken
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    SpacyTextSplitter,
)

from app.models.documents import DocumentChunk
from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""

    chunk_size: int = settings.CHUNK_SIZE
    chunk_overlap: int = settings.CHUNK_OVERLAP
    separators: List[str] = None
    keep_separator: bool = True
    add_start_index: bool = True
    length_function: callable = len


class BaseChunker(ABC):
    """Base class for text chunkers."""

    @abstractmethod
    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text into smaller pieces."""
        pass


class RecursiveCharacterChunker(BaseChunker):
    """Recursive character-based chunker using LangChain."""

    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text recursively by characters."""
        separators = config.separators or [
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            " ",  # Spaces
            "",  # Character level
        ]

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=separators,
            keep_separator=config.keep_separator,
            add_start_index=config.add_start_index,
            length_function=config.length_function,
        )

        return splitter.split_text(text)


class TokenBasedChunker(BaseChunker):
    """Token-based chunker for precise token control."""

    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.encoding = tiktoken.encoding_for_model(model_name)

    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text based on token count."""
        splitter = TokenTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            encoding_name=self.encoding.name,
        )

        return splitter.split_text(text)


class SemanticChunker(BaseChunker):
    """Semantic chunker that tries to preserve meaning."""

    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text semantically using sentence boundaries."""
        # Split by sentences first
        sentences = self._split_sentences(text)

        chunks = []
        current_chunk = ""
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > config.chunk_size and current_chunk:
                chunks.append(current_chunk.strip())

                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(
                    current_chunk, config.chunk_overlap
                )
                current_chunk = overlap_text + sentence
                current_length = len(current_chunk)
            else:
                current_chunk += sentence
                current_length += sentence_length

        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with spaCy
        sentence_pattern = r"(?<=[.!?])\s+"
        sentences = re.split(sentence_pattern, text)
        return [s.strip() + " " for s in sentences if s.strip()]

    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from the end of previous chunk."""
        if len(text) <= overlap_size:
            return text

        # Try to find a good breaking point within overlap size
        overlap_text = text[-overlap_size:]

        # Find the last sentence boundary within overlap
        sentence_end = max(
            overlap_text.rfind("."), overlap_text.rfind("!"), overlap_text.rfind("?")
        )

        if sentence_end > 0:
            return overlap_text[sentence_end + 1 :].strip() + " "

        return overlap_text


class DocumentStructureChunker(BaseChunker):
    """Document structure-aware chunker."""

    def chunk_text(self, text: str, config: ChunkingConfig) -> List[str]:
        """Chunk text based on document structure."""
        # Detect document structure
        sections = self._detect_sections(text)

        chunks = []
        for section in sections:
            # If section is small enough, keep as single chunk
            if len(section) <= config.chunk_size:
                chunks.append(section)
            else:
                # Recursively chunk large sections
                recursive_chunker = RecursiveCharacterChunker()
                section_chunks = recursive_chunker.chunk_text(section, config)
                chunks.extend(section_chunks)

        return chunks

    def _detect_sections(self, text: str) -> List[str]:
        """Detect document sections based on headers and structure."""
        # Simple header detection - can be improved
        lines = text.split("\n")
        sections = []
        current_section = []

        for line in lines:
            # Check if line is a header (starts with #, or is all caps, etc.)
            if self._is_header(line) and current_section:
                sections.append("\n".join(current_section))
                current_section = [line]
            else:
                current_section.append(line)

        if current_section:
            sections.append("\n".join(current_section))

        return sections

    def _is_header(self, line: str) -> bool:
        """Check if line is likely a header."""
        line = line.strip()
        if not line:
            return False

        # Markdown headers
        if line.startswith("#"):
            return True

        # All caps short lines
        if line.isupper() and len(line) < 100:
            return True

        # Lines ending with colon
        if line.endswith(":") and len(line) < 100:
            return True

        return False


class ChunkProcessor:
    """Main chunk processing service."""

    def __init__(self):
        self.chunkers = {
            "recursive": RecursiveCharacterChunker(),
            "token": TokenBasedChunker(),
            "semantic": SemanticChunker(),
            "structure": DocumentStructureChunker(),
        }

    def process_document(
        self,
        document_id: str,
        text: str,
        chunking_strategy: str = "recursive",
        config: Optional[ChunkingConfig] = None,
    ) -> List[DocumentChunk]:
        """Process document text into chunks."""
        if config is None:
            config = ChunkingConfig()

        chunker = self.chunkers.get(chunking_strategy)
        if not chunker:
            raise ValueError(f"Unsupported chunking strategy: {chunking_strategy}")

        try:
            # Chunk the text
            text_chunks = chunker.chunk_text(text, config)

            # Create DocumentChunk objects
            chunks = []
            current_pos = 0

            for i, chunk_text in enumerate(text_chunks):
                # Find the actual position in the original text
                start_pos = text.find(chunk_text, current_pos)
                if start_pos == -1:
                    start_pos = current_pos

                end_pos = start_pos + len(chunk_text)
                current_pos = end_pos

                chunk = DocumentChunk(
                    id=f"{document_id}_chunk_{i}",
                    document_id=document_id,
                    content=chunk_text,
                    chunk_index=i,
                    start_char=start_pos,
                    end_char=end_pos,
                    metadata={
                        "chunking_strategy": chunking_strategy,
                        "chunk_size": config.chunk_size,
                        "chunk_overlap": config.chunk_overlap,
                        "character_count": len(chunk_text),
                        "word_count": len(chunk_text.split()),
                    },
                )
                chunks.append(chunk)

            logger.info(f"Created {len(chunks)} chunks for document {document_id}")
            return chunks

        except Exception as e:
            logger.error(f"Failed to process chunks for document {document_id}: {e}")
            raise

    def get_chunk_statistics(self, chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """Get statistics about the chunks."""
        if not chunks:
            return {}

        chunk_sizes = [len(chunk.content) for chunk in chunks]
        word_counts = [len(chunk.content.split()) for chunk in chunks]

        return {
            "total_chunks": len(chunks),
            "total_characters": sum(chunk_sizes),
            "total_words": sum(word_counts),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "avg_word_count": sum(word_counts) / len(word_counts),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes),
            "chunking_strategy": chunks[0].metadata.get("chunking_strategy", "unknown"),
        }

    def optimize_chunking_config(
        self,
        text: str,
        target_chunk_count: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
    ) -> ChunkingConfig:
        """Optimize chunking configuration based on text characteristics."""
        text_length = len(text)

        if target_chunk_count:
            # Calculate chunk size based on target count
            chunk_size = max(500, text_length // target_chunk_count)
        elif max_chunk_size:
            chunk_size = max_chunk_size
        else:
            # Default adaptive sizing
            if text_length < 5000:
                chunk_size = 1000
            elif text_length < 50000:
                chunk_size = 1500
            else:
                chunk_size = 2000

        overlap = min(200, chunk_size // 5)  # 20% overlap max

        return ChunkingConfig(chunk_size=chunk_size, chunk_overlap=overlap)

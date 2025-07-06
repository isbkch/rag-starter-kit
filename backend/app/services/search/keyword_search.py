"""
Keyword search engine using BM25 algorithm for text-based search.
"""
import logging
import re
import string
from typing import Any, Dict, List, Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

from app.models.search import SearchResponse, SearchResult

logger = logging.getLogger(__name__)


class KeywordSearchEngine:
    """Keyword search engine using BM25 algorithm."""

    def __init__(self):
        self.bm25 = None
        self.documents = []
        self.document_metadata = []
        self.stemmer = PorterStemmer()
        self.stop_words = set()
        self._download_nltk_data()

    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

        try:
            nltk.data.find("corpora/stopwords")
            self.stop_words = set(stopwords.words("english"))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            self.stop_words = set(stopwords.words("english"))

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for BM25 indexing."""
        # Convert to lowercase
        text = text.lower()

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # Tokenize
        tokens = word_tokenize(text)

        # Remove stop words and stem
        tokens = [
            self.stemmer.stem(token)
            for token in tokens
            if token not in self.stop_words and len(token) > 2
        ]

        return tokens

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for keyword search."""
        try:
            self.documents = []
            self.document_metadata = []

            processed_docs = []

            for doc in documents:
                content = doc.get("content", "")
                if not content:
                    continue

                # Preprocess document content
                processed_content = self._preprocess_text(content)
                processed_docs.append(processed_content)

                # Store original document and metadata
                self.documents.append(content)
                self.document_metadata.append(
                    {
                        "id": doc.get("id", ""),
                        "title": doc.get("title", ""),
                        "source": doc.get("source", ""),
                        "metadata": doc.get("metadata", {}),
                        "chunk_index": doc.get("chunk_index", 0),
                    }
                )

            # Create BM25 index
            if processed_docs:
                self.bm25 = BM25Okapi(processed_docs)
                logger.info(
                    f"Indexed {len(processed_docs)} documents for keyword search"
                )
            else:
                logger.warning("No documents to index for keyword search")

        except Exception as e:
            logger.error(f"Error indexing documents for keyword search: {e}")
            raise

    async def search(
        self, query: str, limit: int = 10, min_score: float = 0.0
    ) -> SearchResponse:
        """Perform keyword search using BM25."""
        try:
            if not self.bm25 or not self.documents:
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=0.0,
                    search_type="keyword",
                )

            import time

            start_time = time.time()

            # Preprocess query
            processed_query = self._preprocess_text(query)

            if not processed_query:
                return SearchResponse(
                    query=query,
                    results=[],
                    total_results=0,
                    search_time=time.time() - start_time,
                    search_type="keyword",
                )

            # Get BM25 scores
            scores = self.bm25.get_scores(processed_query)

            # Get top results
            top_indices = sorted(
                range(len(scores)), key=lambda i: scores[i], reverse=True
            )[
                : limit * 2
            ]  # Get more to filter by min_score

            results = []
            for idx in top_indices:
                score = scores[idx]
                if score < min_score:
                    continue

                metadata = self.document_metadata[idx]
                content = self.documents[idx]

                # Extract context around query terms
                context = self._extract_context(content, query)

                result = SearchResult(
                    content=content,
                    score=float(score),
                    metadata=metadata,
                    source=metadata.get("source", ""),
                    title=metadata.get("title", ""),
                    context=context,
                    citations=self._extract_citations(content, metadata),
                )
                results.append(result)

                if len(results) >= limit:
                    break

            search_time = time.time() - start_time

            return SearchResponse(
                query=query,
                results=results,
                total_results=len(results),
                search_time=search_time,
                search_type="keyword",
            )

        except Exception as e:
            logger.error(f"Error performing keyword search: {e}")
            return SearchResponse(
                query=query,
                results=[],
                total_results=0,
                search_time=0.0,
                search_type="keyword",
                error=str(e),
            )

    def _extract_context(
        self, content: str, query: str, context_length: int = 200
    ) -> str:
        """Extract context around query terms."""
        try:
            query_terms = query.lower().split()
            content_lower = content.lower()

            # Find the first occurrence of any query term
            best_position = -1
            for term in query_terms:
                position = content_lower.find(term)
                if position != -1:
                    if best_position == -1 or position < best_position:
                        best_position = position

            if best_position == -1:
                # No query terms found, return beginning of content
                return content[:context_length] + (
                    "..." if len(content) > context_length else ""
                )

            # Extract context around the found position
            start = max(0, best_position - context_length // 2)
            end = min(len(content), best_position + context_length // 2)

            context = content[start:end]

            # Add ellipsis if we're not at the beginning/end
            if start > 0:
                context = "..." + context
            if end < len(content):
                context = context + "..."

            return context

        except Exception as e:
            logger.error(f"Error extracting context: {e}")
            return content[:context_length] + (
                "..." if len(content) > context_length else ""
            )

    def _extract_citations(self, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Extract citations from content and metadata."""
        citations = []

        # Add source as citation
        source = metadata.get("source", "")
        if source:
            citations.append(source)

        # Add title as citation if different from source
        title = metadata.get("title", "")
        if title and title != source:
            citations.append(title)

        return citations

    def get_stats(self) -> Dict[str, Any]:
        """Get keyword search engine statistics."""
        return {
            "indexed_documents": len(self.documents),
            "has_index": self.bm25 is not None,
            "engine_type": "BM25Okapi",
        }

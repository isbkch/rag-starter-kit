# ADR-001: Use of Hybrid Search over Vector-Only Search

* **Status**: Accepted
* **Date**: 2025-07-06
* **Deciders**: Ilyas Bakouch, Team

## Context and Problem Statement

Retrieval-Augmented Generation (RAG) systems require a robust method for retrieving relevant context to answer user queries. While vector search is powerful for finding semantically similar content, it can sometimes miss documents that contain exact keyword matches but have a different semantic structure. For example, a query for a specific product ID or a technical term might not be well-served by vector search alone if the surrounding context is not semantically similar to the query.

## Decision Drivers

* **Improved Retrieval Accuracy**: Combining keyword search with vector search can lead to more accurate and comprehensive retrieval results.
* **Handling of Specific Terms**: Keyword search excels at finding documents containing specific, literal terms (e.g., SKUs, error codes, acronyms) that vector search might overlook.
* **User Expectation**: Users often expect systems to find exact matches for the terms they enter.

## Considered Options

1. **Vector Search Only**: Rely solely on the vector database for retrieval. This is simpler to implement but may have the weaknesses described above.
2. **Keyword Search Only**: Rely solely on a keyword search engine like Elasticsearch. This would lose the benefits of semantic search.
3. **Hybrid Search**: Combine both vector and keyword search, and re-rank the results to get the best of both worlds.

## Decision Outcome

Chosen option: **Hybrid Search**. We will implement a hybrid search system that queries both a vector database and a keyword search engine (Elasticsearch) in parallel. The results will be merged and re-ranked using a technique like Reciprocal Rank Fusion (RRF) to produce a final, high-quality set of context documents.

### Positive Consequences

* Improved search relevance and user satisfaction.
* Better handling of a wider variety of query types.
* The system is more robust and less prone to the weaknesses of a single search method.

### Negative Consequences

* Increased system complexity, as it requires managing and querying two different search systems.
* Slightly higher latency due to the need to perform two searches and a re-ranking step.
* Increased infrastructure cost due to the need for both a vector database and an Elasticsearch cluster.

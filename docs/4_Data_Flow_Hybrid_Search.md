# 4. Data Flow - Hybrid Search & Retrieval

This document outlines the data flow for the hybrid search and retrieval process. This workflow is triggered when a user submits a query and is designed to return the most relevant document chunks to be used as context for a Large Language Model (LLM).

## Search Flow Diagram

![Search Flow](https://placehold.co/1200x600?text=Hybrid%20Search%20%26%20Retrieval%20Flow%0A(Sequence%20Diagram))
*A sequence diagram showing how a user query is processed to retrieve relevant context.*

## Step-by-Step Process

1. **API Request**: The frontend sends a `POST` request to the `/api/v1/search` endpoint on the **Backend API**. The request payload contains the user's query string.

2. **Query Processing (Backend API)**:
    * The API receives the query and hands it off to the `SearchManager` service.
    * The `SearchManager` orchestrates the hybrid search by initiating two parallel search processes: **Vector Search** and **Keyword Search**.

3. **Parallel Search Execution**:
    * **Vector Search**:
        * The `EmbeddingService` is called to convert the user's query into a vector embedding using the **OpenAI API**.
        * The `VectorSearchService` takes this query embedding and searches the active **Vector Database** (e.g., ChromaDB) to find the top-k most semantically similar document chunks.
    * **Keyword Search**:
        * The `KeywordSearchService` takes the raw query string and performs a BM25 search against the **Elasticsearch** index to find the top-k most relevant document chunks based on keyword matching.

4. **Re-ranking and Merging**:
    * The results from both the vector search and keyword search are collected by the `HybridSearchService`.
    * The service then applies a re-ranking algorithm to combine the two sets of results. This typically involves a weighted scoring system (e.g., Reciprocal Rank Fusion) to produce a single, unified list of the most relevant document chunks.
    * Duplicate chunks are removed, and the final list is truncated to a predefined number of results.

5. **Context Generation & LLM Interaction**:
    * The re-ranked and filtered list of document chunks is compiled into a single block of text. This text serves as the **context** for the LLM.
    * The original user query and the prepared context are sent to the configured **OpenAI LLM** (e.g., GPT-4) with a prompt that instructs the model to answer the user's question based on the provided context.

6. **API Response**:
    * The Backend API receives the generated answer from the LLM.
    * It formats the final response, which may include the answer, the source document chunks, and other metadata.
    * This response is sent back to the frontend to be displayed to the user.

## Key Services and Modules Involved

* **API Endpoint**: `app/api/v1/endpoints/search.py`
* **Search Orchestrator**: `app/services/search/search_manager.py`
* **Hybrid Search Logic**: `app/services/search/hybrid_search.py`
* **Vector Search**: `app/services/search/vector_search.py`
* **Keyword Search**: `app/services/search/keyword_search.py`
* **Embedding Service**: `app/services/search/embedding_service.py`

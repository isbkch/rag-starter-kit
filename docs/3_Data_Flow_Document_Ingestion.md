# 3. Data Flow - Document Ingestion

This document describes the end-to-end data flow for the document ingestion process in the Enterprise RAG Platform. This process is designed to be asynchronous, scalable, and robust.

## Ingestion Flow Diagram

![Ingestion Flow](https://placehold.co/1200x600?text=Document%20Ingestion%20Data%20Flow%0A(Sequence%20Diagram))
*A sequence diagram illustrating the steps from document upload to final storage.*

## Step-by-Step Process

The ingestion process begins when a user uploads a document via the frontend.

1. **API Request**: The frontend sends a `POST` request to the `/api/v1/documents` endpoint on the **Backend API**. The request includes the file and any associated metadata.

2. **Initial Handling (Backend API)**:
    * The API receives the request and performs initial validation.
    * It creates a new document record in the **PostgreSQL** database with a `status` of `PENDING`.
    * It then schedules an asynchronous ingestion task by sending a message to the **Celery** task queue via **Redis**.
    * The API immediately returns a response to the user with a `202 Accepted` status and the document ID, indicating that the ingestion has started.

3. **Asynchronous Processing (Celery Worker)**:
    * A **Celery worker** picks up the ingestion task from the queue.
    * The worker updates the document status in **PostgreSQL** to `PROCESSING`.

4. **Document Processing Pipeline**:
    * **Text Extraction**: The worker uses the `TextExtractor` to extract raw text content from the document (supporting PDF, DOCX, MD, and TXT).
    * **Chunking**: The extracted text is passed to the `ChunkProcessor`, which splits the text into smaller, semantically meaningful chunks. The chunking strategy is configurable.
    * **Embedding Generation**: For each chunk, the `EmbeddingService` is called. This service communicates with the configured **OpenAI API** to generate a vector embedding for the chunk's content.

5. **Data Storage**:
    * **Vector Database**: The generated embeddings, along with the chunk content and metadata, are stored in the active **Vector Database** (e.g., ChromaDB, Pinecone). This enables semantic search.
    * **Keyword Search Index**: The raw text of the chunks is indexed in **Elasticsearch**. This enables keyword-based (BM25) search.

6. **Finalization**:
    * Once all chunks are successfully processed and stored, the Celery worker updates the document's status in **PostgreSQL** to `COMPLETED`.
    * If any step in the pipeline fails, the status is updated to `FAILED`, and the error is logged for debugging.

## Key Services and Modules Involved

* **API Endpoint**: `app/api/v1/endpoints/documents.py`
* **Celery Task**: Defined in `app/services/ingestion/document_processor.py`
* **Text Extractor**: `app/services/ingestion/text_extractor.py`
* **Chunk Processor**: `app/services/ingestion/chunk_processor.py`
* **Embedding Service**: `app/services/search/embedding_service.py`
* **Vector DB Clients**: `app/services/vectordb/`

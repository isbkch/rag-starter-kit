# 1. Architecture Overview

This document provides a high-level overview of the Enterprise RAG Platform architecture. It is intended for developers, architects, and operations personnel who need to understand the system's structure and key components.

## System Components

The platform is designed as a set of containerized services that work together to provide a production-ready Retrieval-Augmented Generation (RAG) solution.

![Architecture Diagram](https://placehold.co/800x500?text=Architecture%20Diagram%0A(System%20Context%20View))
*A high-level diagram illustrating the interaction between the core components.*

The primary components are:

* **Frontend**: A React-based single-page application (SPA) that provides the user interface for document management and search. It communicates with the Backend API.
* **Backend API**: A FastAPI application that serves as the primary entry point for the system. It handles user authentication, document uploads, search queries, and orchestrates the various backend services.
* **Celery Worker & Beat**: Asynchronous task queue for handling long-running processes, primarily document ingestion and processing. Celery Beat is used for scheduled tasks.
* **PostgreSQL Database**: The primary relational database for storing structured data, such as document metadata, user information, and job status.
* **Redis**: An in-memory data store used for caching API responses, managing Celery task queues, and implementing rate limiting.
* **Vector Database (ChromaDB/Pinecone/Weaviate)**: A specialized database for storing vector embeddings of document chunks. It enables efficient semantic similarity search. The system uses a factory pattern to allow for pluggable providers.
* **Elasticsearch**: A search engine used for keyword-based (BM25) search, complementing the vector search to provide a hybrid search capability.
* **Monitoring Stack**: Prometheus for metrics collection and Grafana for visualization, providing observability into the system's health and performance.

## Core Workflows

1. **Document Ingestion**: Users upload documents through the frontend. The Backend API creates a record in PostgreSQL and schedules an asynchronous ingestion task with Celery. The Celery worker then processes the document, extracts text, chunks it, generates embeddings, and stores the data in both the Vector Database and Elasticsearch.
2. **Hybrid Search**: Users submit a query through the frontend. The Backend API simultaneously queries the Vector Database for semantic matches and Elasticsearch for keyword matches. The results are combined and re-ranked to produce the most relevant context, which is then passed to a Large Language Model (LLM) to generate a final answer.

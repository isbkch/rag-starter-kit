# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the Enterprise RAG Platform Starter Kit - a production-ready Retrieval-Augmented Generation platform with enterprise-grade features. It combines vector similarity search with BM25 keyword search, supports multiple vector database providers (Pinecone, Weaviate, ChromaDB), and includes comprehensive monitoring and scalability features.

## Technology Stack

- **Backend**: FastAPI (Python 3.13+), SQLAlchemy, Redis, Celery
- **Frontend**: React 19.1.0, TypeScript, Vite, Tailwind CSS
- **Vector Databases**: ChromaDB, Pinecone, Weaviate
- **Search**: LangChain, OpenAI embeddings, Elasticsearch, FAISS
- **Infrastructure**: Docker Compose, PostgreSQL, Redis, Elasticsearch

## Common Development Commands

### Backend Development

```bash
# Start development server
cd backend && uvicorn app.main:app --reload

# Run tests with coverage
cd backend && pytest --cov=app --cov-report=html

# Run specific test types
cd backend && pytest -m unit          # Unit tests only
cd backend && pytest -m integration   # Integration tests only
cd backend && pytest -m performance   # Performance tests only

# Code formatting and linting
cd backend && black .
cd backend && isort .
cd backend && flake8
cd backend && mypy .
```

### Frontend Development

```bash
# Start development server
cd frontend && npm run dev

# Build for production
cd frontend && npm run build

# Run linting
cd frontend && npm run lint

# Preview production build
cd frontend && npm run preview
```

### Docker Environment

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Rebuild services
docker-compose up --build
```

## Architecture Overview

The platform follows a microservices architecture with clear separation of concerns:

- **`backend/app/api/v1/`**: REST API endpoints (documents, search, health)
- **`backend/app/core/`**: Core configuration, authentication, rate limiting
- **`backend/app/models/`**: Database models using SQLAlchemy
- **`backend/app/services/`**: Business logic layer
  - `ingestion/`: Document processing pipeline
  - `search/`: Hybrid search engines (vector + keyword)
  - `vectordb/`: Vector database abstraction layer
- **`frontend/src/`**: React components and application logic

## Key Configuration

### Environment Variables (create from .env.example)

- `SECRET_KEY`: Application secret key
- `DATABASE_URL`: PostgreSQL connection string
- `VECTOR_DB_PROVIDER`: chroma/pinecone/weaviate
- `OPENAI_API_KEY`: For embeddings and LLM services
- Provider-specific keys for Pinecone/Weaviate if used

### Database Migrations

```bash
# Generate new migration
cd backend && alembic revision --autogenerate -m "description"

# Apply migrations
cd backend && alembic upgrade head
```

## Testing Strategy

The test suite uses pytest with async support and includes:

- **Unit tests**: Fast, isolated component testing
- **Integration tests**: Multi-component interactions
- **Performance tests**: Latency and throughput benchmarks
- **External service tests**: Vector databases, OpenAI API

Test markers are configured in `backend/pytest.ini` for selective test execution.

## Vector Database Abstraction

The platform uses a factory pattern for vector database providers:

- Switch providers via `VECTOR_DB_PROVIDER` environment variable
- Each provider implements the same interface in `backend/app/services/vectordb/`
- Configuration is centralized in `backend/app/core/config.py`

## Document Processing Pipeline

Multi-format support with intelligent chunking strategies:

- **Formats**: PDF, DOCX, Markdown, plain text
- **Chunking**: Recursive, semantic, structure-aware, token-based
- **Processing**: Async with Celery background tasks

## Search System

Hybrid search combining:

- **Vector search**: Semantic similarity using embeddings
- **Keyword search**: BM25 via Elasticsearch
- **Hybrid ranking**: Weighted combination of both approaches

## Monitoring & Observability

Built-in monitoring with:

- OpenTelemetry for distributed tracing
- Prometheus metrics collection
- Structured logging with correlation IDs
- Health check endpoints at `/health` and `/api/v1/health/detailed`

## Production Considerations

- Rate limiting with token bucket algorithm
- Multi-level caching (Redis, in-memory)
- Connection pooling for databases
- Async processing with Celery
- Comprehensive error handling and recovery

## Development Workflow

1. Use Docker Compose for local development environment
2. Backend runs on port 8000, frontend on port 5173
3. API documentation available at `/docs` (Swagger) and `/redoc`
4. Tests should be run before committing changes
5. Follow existing code patterns and conventions

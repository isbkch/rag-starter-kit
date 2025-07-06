# Enterprise RAG Platform Starter Kit

A production-ready Retrieval-Augmented Generation (RAG) platform with enterprise-grade features, hybrid search capabilities, and comprehensive scalability considerations.

## 🚀 Features

### Core Capabilities

- **Multi-format Document Ingestion**: Support for PDF, DOCX, Markdown, and plain text
- **Hybrid Search Engine**: Combines vector similarity and BM25 keyword search
- **Vector Database Abstraction**: Seamless switching between Pinecone, Weaviate, and ChromaDB
- **Production-Ready FastAPI Backend**: Streaming responses, rate limiting, monitoring
- **Modern React Frontend**: Citation highlighting and real-time search (coming soon)
- **Docker Compose Setup**: Complete local development environment
- **Performance Benchmarking**: Built-in latency and throughput metrics
- **AWS Deployment Scripts**: Terraform/CDK for production deployment (coming soon)

### Enterprise Features

- **Multi-provider Vector Database Support**: Choose the best database for your needs
- **Advanced Text Chunking**: Multiple strategies including semantic and structure-aware
- **Comprehensive Monitoring**: OpenTelemetry, Prometheus, and custom metrics
- **Rate Limiting**: Token bucket algorithm with Redis backend
- **Caching Layers**: Multi-level caching for embeddings and search results
- **Async Processing**: Celery for background document processing
- **Error Handling**: Comprehensive error tracking and recovery

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React Frontend │    │  FastAPI Backend │    │  Vector Database │
│                 │────│                 │────│  (Pinecone/     │
│ - Search UI     │    │ - REST API      │    │   Weaviate/     │
│ - Citations     │    │ - Streaming     │    │   ChromaDB)     │
│ - File Upload   │    │ - Rate Limiting │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │  Search Engine  │
                       │                 │
                       │ - Hybrid Search │
                       │ - Vector Search │
                       │ - Keyword Search│
                       └─────────────────┘
                                │
                       ┌─────────────────┐
                       │ Document Pipeline│
                       │                 │
                       │ - Text Extract  │
                       │ - Chunking      │
                       │ - Embeddings    │
                       └─────────────────┘
```

## 🛠️ Technology Stack

### Backend

- **FastAPI**: High-performance async web framework
- **LangChain**: LLM application framework
- **Pydantic**: Data validation and settings management
- **SQLAlchemy**: Database ORM
- **Celery**: Distributed task queue
- **Redis**: Caching and message broker

### Vector Databases

- **ChromaDB**: Open-source embedding database
- **Pinecone**: Managed vector database service
- **Weaviate**: Open-source vector database with GraphQL

### Document Processing

- **PyPDF2/pdfplumber**: PDF text extraction
- **python-docx**: DOCX processing
- **python-markdown**: Markdown parsing
- **tiktoken**: Token counting and chunking

### Search & Embeddings

- **OpenAI Embeddings**: Text-embedding-ada-002
- **Sentence Transformers**: Local embedding models
- **Elasticsearch**: BM25 keyword search
- **FAISS**: Local vector search

### Monitoring & Observability

- **OpenTelemetry**: Distributed tracing
- **Prometheus**: Metrics collection
- **Grafana**: Visualization (coming soon)
- **Structlog**: Structured logging

## 📦 Installation

### Prerequisites

- Python 3.9+
- Node.js 20+
- Docker & Docker Compose
- Redis
- PostgreSQL

### Quick Start

1. **Clone the repository**

   ```bash
   git clone git@github.com:isbkch/rag-starter-kit.git
   cd rag-starter-kit
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install Python dependencies**

   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Start with Docker Compose**

   ```bash
   docker-compose up -d
   ```

5. **Install frontend dependencies**

   ```bash
   cd frontend
   npm install
   ```

6. **Run the backend**

   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

7. **Run the frontend** (in a new terminal)

   ```bash
   cd frontend
   npm run dev
   ```

8. **Access the application**
   - Frontend: http://localhost:5173
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## ⚙️ Configuration

### Environment Variables

```bash
# Application
DEBUG=false
SECRET_KEY=your-secret-key-here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/rag_platform
REDIS_URL=redis://localhost:6379

# Vector Database (choose one)
VECTOR_DB_PROVIDER=chroma  # Options: chroma, pinecone, weaviate

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east1-gcp
PINECONE_INDEX_NAME=rag-platform

# Weaviate
WEAVIATE_URL=http://localhost:8080
WEAVIATE_API_KEY=your-weaviate-api-key

# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Search Configuration
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_DIMENSIONS=1536
MAX_SEARCH_RESULTS=10
SIMILARITY_THRESHOLD=0.7

# Document Processing
MAX_FILE_SIZE=52428800  # 50MB
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

## 🔌 API Endpoints

### Document Management

- `POST /api/v1/documents/upload` - Upload document for processing
- `GET /api/v1/documents` - List all documents with pagination
- `GET /api/v1/documents/{id}` - Get document details
- `DELETE /api/v1/documents/{id}` - Delete document
- `POST /api/v1/documents/{id}/process` - Trigger document processing

### Search

- `POST /api/v1/search` - Perform hybrid search
- `POST /api/v1/search/vector` - Vector similarity search only
- `POST /api/v1/search/keyword` - Keyword search only
- `GET /api/v1/search/suggestions` - Get search suggestions

### Health & Monitoring

- `GET /health` - Basic health check
- `GET /api/v1/health/detailed` - Detailed system health
- `GET /api/v1/metrics` - Performance metrics

## 📋 Technical Specifications

### System Requirements

- **CPU**: 4+ cores recommended for production
- **Memory**: 8GB RAM minimum, 16GB+ for large document sets
- **Storage**: 100GB+ SSD for vector indexes and document storage
- **Network**: 1Gbps for high-throughput deployments

### Performance Benchmarks

- **Search Latency**: < 100ms for vector search (p95)
- **Ingestion Rate**: 50+ documents/minute (varies by size)
- **Concurrent Users**: 100+ simultaneous search requests
- **Vector Database**: Supports 1M+ documents with sub-second search

### Supported Document Formats

| Format | Max Size | Processing Time | Notes |
|--------|----------|----------------|--------|
| PDF | 50MB | ~30s | OCR supported for scanned PDFs |
| DOCX | 25MB | ~15s | Full formatting preservation |
| Markdown | 10MB | ~5s | Native support with metadata |
| Plain Text | 10MB | ~3s | Fastest processing |

### Vector Database Comparison

| Provider | Pros | Cons | Best For |
|----------|------|------|----------|
| **ChromaDB** | Open source, local deployment | Limited scaling | Development, small teams |
| **Pinecone** | Managed service, excellent performance | Cost, vendor lock-in | Production, enterprise |
| **Weaviate** | GraphQL, semantic search | Complexity | Advanced use cases |

## 🔍 Usage

### Document Ingestion

The platform supports multiple document formats with intelligent text extraction:

```python
from app.services.ingestion import DocumentProcessor

processor = DocumentProcessor()

# Process a document
document = await processor.process_document(
    file_path="path/to/document.pdf",
    original_filename="document.pdf",
    chunking_strategy="recursive",  # or "semantic", "structure", "token"
)
```

### Search

Perform hybrid search combining vector similarity and keyword matching:

```python
from app.services.search import HybridSearchEngine

search_engine = HybridSearchEngine()

results = await search_engine.search(
    query="What is machine learning?",
    search_type="hybrid",  # or "vector", "keyword"
    max_results=10,
    similarity_threshold=0.7,
)
```

### Vector Database Management

Switch between vector database providers seamlessly:

```python
from app.services.vectordb import VectorDBFactory

# Create a vector database instance
vector_db = VectorDBFactory.create_vector_db(
    provider="pinecone",  # or "chroma", "weaviate"
)

await vector_db.connect()
```

## 📊 Performance & Scaling

### Benchmarking

Built-in performance benchmarking suite:

```bash
cd backend
python -m app.benchmarks.run_benchmarks
```

### Scaling to 1M Queries

The platform is designed to handle enterprise-scale workloads:

1. **Horizontal Scaling**: Multi-region deployment with load balancers
2. **Caching Strategy**: Multi-layer caching (Redis, in-memory, CDN)
3. **Database Optimization**: Connection pooling, read replicas
4. **Async Processing**: Background task processing with Celery
5. **Auto-scaling**: Kubernetes HPA based on CPU/memory/custom metrics

### Cost Optimization

- **Tiered Storage**: Hot/warm/cold data strategies
- **Spot Instances**: For batch processing workloads
- **Vector Quantization**: Reduce storage and computation costs
- **Efficient Indexing**: Approximate nearest neighbor algorithms
- **Caching**: Reduce expensive LLM API calls

## 🚀 Deployment

### Local Development

```bash
docker-compose up -d
```

### Production (AWS)

Terraform scripts for AWS deployment:

```bash
cd infrastructure/terraform
terraform init
terraform plan
terraform apply
```

### Kubernetes

```bash
cd infrastructure/k8s
kubectl apply -f .
```

## 📈 Monitoring

### Health Checks

- `/health` - Basic health check
- `/api/v1/health/detailed` - Detailed service status

### Metrics

- Request latency and throughput
- Search performance metrics
- Vector database statistics
- Embedding cache hit rates
- Error rates and types

### Logging

Structured logging with correlation IDs for distributed tracing.

## 🧪 Testing

### Backend Tests

```bash
cd backend

# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test types
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m performance   # Performance tests only

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_search.py
```

### Frontend Tests

```bash
cd frontend

# Run unit tests
npm test

# Run tests with coverage
npm run test:coverage

# Run E2E tests
npm run test:e2e
```

## 🔧 Development Workflow

### Code Quality

```bash
# Backend formatting and linting
cd backend
black .                 # Format code
isort .                 # Sort imports
flake8                  # Lint code
mypy .                  # Type checking

# Frontend linting
cd frontend
npm run lint           # ESLint
npm run lint:fix       # Auto-fix issues
npm run type-check     # TypeScript checking
```

### Database Migrations

```bash
cd backend

# Create new migration
alembic revision --autogenerate -m "description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Adding New Vector Database Provider

1. Create provider class in `backend/app/services/vectordb/providers/`
2. Implement the `VectorDBInterface` abstract methods
3. Add provider to factory in `backend/app/services/vectordb/factory.py`
4. Update configuration in `backend/app/core/config.py`
5. Add tests in `backend/tests/services/vectordb/`

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
cd backend
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

## 🚨 Troubleshooting

### Common Issues

#### Backend Issues

**Issue**: `ImportError: No module named 'app'`
```bash
# Solution: Set PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)/backend"
# Or run from backend directory
cd backend && uvicorn app.main:app --reload
```

**Issue**: Vector database connection fails
```bash
# Check vector database is running
docker-compose ps

# Check environment variables
echo $VECTOR_DB_PROVIDER
echo $CHROMA_HOST  # or other provider variables

# Verify connectivity
curl http://localhost:8000/api/v1/health/detailed
```

**Issue**: OpenAI API key errors
```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test API key
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"
```

#### Frontend Issues

**Issue**: API calls fail with CORS errors
```bash
# Check if backend is running
curl http://localhost:8000/health

# Verify proxy configuration in vite.config.ts
# Ensure proxy target matches backend URL
```

**Issue**: `npm install` fails
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

#### Docker Issues

**Issue**: Services fail to start
```bash
# Check logs
docker-compose logs -f [service-name]

# Rebuild services
docker-compose down
docker-compose up --build

# Check disk space
df -h
```

**Issue**: Port conflicts
```bash
# Check what's using the port
lsof -i :8000  # Backend port
lsof -i :5173  # Frontend port

# Kill process or change port in configuration
```

### Performance Issues

**Issue**: Slow search responses
1. Check vector database performance: `GET /api/v1/health/detailed`
2. Monitor embedding cache hit rate
3. Verify similarity threshold isn't too low
4. Consider reducing `max_results` parameter

**Issue**: High memory usage
1. Monitor embedding cache size
2. Reduce `CHUNK_SIZE` for large documents
3. Implement document cleanup for old files
4. Consider using vector quantization

### Debugging Tips

```bash
# Enable debug logging
export DEBUG=true

# Monitor API requests
tail -f backend/logs/app.log

# Check database connections
export DATABASE_URL=postgresql://user:pass@localhost:5432/dbname
python -c "from sqlalchemy import create_engine; create_engine('$DATABASE_URL').connect()"

# Test vector database directly
python -c "
from app.services.vectordb.factory import VectorDBFactory
db = VectorDBFactory.create_vector_db()
print(db.health_check())
"
```

### Getting Help

1. **Check logs**: Always start with application and container logs
2. **API Documentation**: Visit `/docs` for interactive API testing
3. **Health Endpoints**: Use `/health` and `/api/v1/health/detailed`
4. **GitHub Issues**: Search existing issues or create new ones
5. **Configuration**: Verify all required environment variables are set

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Documentation](docs/)
- [API Reference](http://localhost:8000/docs)
- [Architecture Guide](docs/architecture/)
- [Deployment Guide](docs/deployment/)
- [Scaling Guide](docs/scaling/)

## 🆘 Support

For support and questions:

- Create an issue in the repository
- Check the documentation in the `docs/` folder
- Review the example configurations

---

**Enterprise RAG Platform Starter Kit** - Production-ready RAG with enterprise features and comprehensive scalability planning.

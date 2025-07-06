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

5. **Run the backend**

   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

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

```bash
cd backend
pytest
```

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

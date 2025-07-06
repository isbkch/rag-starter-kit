"""
Pytest configuration and fixtures for the RAG platform tests.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from app.core.config import Settings
from app.models.database import Base
from app.services.vectordb.base import VectorDBConfig
from app.services.search.embedding_service import EmbeddingService


# Test database URL (in-memory SQLite)
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


class TestSettings(Settings):
    """Test settings override."""

    DATABASE_URL: str = TEST_DATABASE_URL
    REDIS_URL: str = "redis://localhost:6379/15"  # Use different DB for tests

    # Test OpenAI settings
    OPENAI_API_KEY: str = "test-openai-key"

    # Vector DB settings for tests
    VECTOR_DB_PROVIDER: str = "chroma"
    VECTOR_DB_COLLECTION: str = "test_collection"

    # Disable rate limiting in tests
    ENABLE_RATE_LIMITING: bool = False

    # Test-specific settings
    TESTING: bool = True
    LOG_LEVEL: str = "DEBUG"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> TestSettings:
    """Test settings fixture."""
    return TestSettings()


@pytest_asyncio.fixture
async def async_engine():
    """Create async database engine for tests."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
        echo=False,
    )

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create database session for tests."""
    async_session = sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )

    async with async_session() as session:
        yield session


@pytest.fixture
def client() -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def mock_redis():
    """Mock Redis client."""
    mock_redis = AsyncMock()
    mock_redis.ping.return_value = True
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.incr.return_value = 1
    mock_redis.expire.return_value = True
    mock_redis.pipeline.return_value.__aenter__.return_value.execute.return_value = [
        1,
        True,
    ]
    return mock_redis


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client."""
    mock_client = Mock()
    mock_embeddings = Mock()
    mock_embeddings.create.return_value.data = [
        Mock(embedding=[0.1, 0.2, 0.3] * 512)  # 1536 dimensions
    ]
    mock_client.embeddings = mock_embeddings
    return mock_client


@pytest.fixture
def mock_vector_db():
    """Mock vector database."""
    mock_db = AsyncMock()
    mock_db.connect.return_value = None
    mock_db.disconnect.return_value = None
    mock_db.insert_vectors.return_value = ["test-id-1", "test-id-2"]
    mock_db.search_vectors.return_value = []
    mock_db.get_collection_stats.return_value = {"count": 0}
    return mock_db


@pytest.fixture
def vector_db_config() -> VectorDBConfig:
    """Vector database configuration for tests."""
    return VectorDBConfig(
        provider="chroma",
        collection_name="test_collection",
        embedding_dimension=1536,
        distance_metric="cosine",
    )


@pytest_asyncio.fixture
async def embedding_service(mock_redis, mock_openai_client) -> EmbeddingService:
    """Create embedding service for tests."""
    service = EmbeddingService()
    service.redis_client = mock_redis
    service.openai_client = mock_openai_client
    return service


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        {
            "id": "doc1",
            "title": "Introduction to Machine Learning",
            "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            "metadata": {"author": "John Doe", "category": "AI"},
        },
        {
            "id": "doc2",
            "title": "Deep Learning Fundamentals",
            "content": "Deep learning uses neural networks with multiple layers to model and understand complex patterns in data.",
            "metadata": {"author": "Jane Smith", "category": "Deep Learning"},
        },
    ]


@pytest.fixture
def sample_chunks():
    """Sample document chunks for testing."""
    return [
        {
            "chunk_id": "chunk1",
            "document_id": "doc1",
            "content": "Machine learning is a subset of artificial intelligence.",
            "chunk_index": 0,
            "start_char": 0,
            "end_char": 55,
            "metadata": {"document_title": "Introduction to Machine Learning"},
        },
        {
            "chunk_id": "chunk2",
            "document_id": "doc1",
            "content": "It focuses on algorithms that can learn from data.",
            "chunk_index": 1,
            "start_char": 56,
            "end_char": 104,
            "metadata": {"document_title": "Introduction to Machine Learning"},
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Sample embeddings for testing."""
    return [
        [0.1, 0.2, 0.3] * 512,  # 1536 dimensions
        [0.4, 0.5, 0.6] * 512,  # 1536 dimensions
    ]


@pytest.fixture
def sample_search_query():
    """Sample search query for testing."""
    return {
        "query": "What is machine learning?",
        "search_type": "hybrid",
        "max_results": 10,
        "similarity_threshold": 0.7,
    }


# Async test helpers
@pytest_asyncio.fixture
async def setup_test_data(db_session, sample_documents):
    """Set up test data in the database."""
    from app.models.database import Document, DocumentChunk

    # Add documents
    for doc_data in sample_documents:
        document = Document(
            id=doc_data["id"],
            title=doc_data["title"],
            content=doc_data["content"],
            file_type="text",
            file_size=len(doc_data["content"]),
            metadata_=doc_data["metadata"],
        )
        db_session.add(document)

    await db_session.commit()
    return sample_documents


# Test utilities
class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


def assert_valid_embedding(embedding, expected_dimension=1536):
    """Assert that an embedding is valid."""
    assert isinstance(embedding, list)
    assert len(embedding) == expected_dimension
    assert all(isinstance(x, (int, float)) for x in embedding)


def assert_valid_search_result(result):
    """Assert that a search result is valid."""
    required_fields = ["id", "score", "content"]
    for field in required_fields:
        assert field in result

    assert isinstance(result["score"], (int, float))
    assert 0 <= result["score"] <= 1
    assert isinstance(result["content"], str)


# Performance testing utilities
@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing."""
    return {
        "embedding_generation": 2.0,  # seconds
        "vector_search": 1.0,  # seconds
        "document_ingestion": 5.0,  # seconds
    }


# Mock external services
@pytest.fixture
def mock_external_services(monkeypatch, mock_redis, mock_openai_client):
    """Mock all external services."""
    # Mock Redis
    monkeypatch.setattr("app.services.search.embedding_service.redis", mock_redis)

    # Mock OpenAI
    monkeypatch.setattr(
        "app.services.search.embedding_service.openai.OpenAI",
        lambda **kwargs: mock_openai_client,
    )

    # Mock vector databases
    from app.services.vectordb import factory

    monkeypatch.setattr(factory, "create_vector_db", lambda **kwargs: mock_vector_db())


# Cleanup fixtures
@pytest_asyncio.fixture
async def cleanup_redis(mock_redis):
    """Cleanup Redis after tests."""
    yield
    if hasattr(mock_redis, "flushdb"):
        await mock_redis.flushdb()


@pytest.fixture(autouse=True)
def cleanup_files():
    """Cleanup temporary files after tests."""
    import tempfile
    import shutil

    temp_dirs = []

    def create_temp_dir():
        temp_dir = tempfile.mkdtemp()
        temp_dirs.append(temp_dir)
        return temp_dir

    yield create_temp_dir

    # Cleanup
    for temp_dir in temp_dirs:
        try:
            shutil.rmtree(temp_dir)
        except Exception:
            pass


# Test markers
pytest_plugins = []


# Custom markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow tests")
    config.addinivalue_line("markers", "external: Tests that require external services")

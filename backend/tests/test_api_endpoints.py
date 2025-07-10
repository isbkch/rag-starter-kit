"""
Tests for API endpoints.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.mark.unit
class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_basic_health_check(self, client: TestClient):
        """Test basic health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "app_name" in data
        assert "version" in data
        assert "timestamp" in data

    def test_detailed_health_check(self, client: TestClient):
        """Test detailed health check endpoint."""
        with patch("app.api.v1.endpoints.health.detailed_health_check") as mock_health:
            mock_health.return_value = {
                "status": "healthy",
                "database": "connected",
                "redis": "connected",
                "vector_db": "connected",
            }

            response = client.get("/api/v1/health/detailed")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "database" in data
            assert "redis" in data
            assert "vector_db" in data


@pytest.mark.unit
class TestSearchEndpoints:
    """Test search API endpoints."""

    @patch("app.services.search.search_manager.SearchManager.search")
    def test_search_endpoint(self, mock_search, client: TestClient):
        """Test search endpoint."""
        # Mock search results
        mock_search.return_value = {
            "results": [
                {
                    "id": "doc1",
                    "score": 0.9,
                    "content": "Machine learning content",
                    "metadata": {"title": "ML Guide"},
                }
            ],
            "total": 1,
            "query_time": 0.1,
        }

        search_request = {
            "query": "machine learning",
            "search_type": "hybrid",
            "max_results": 10,
        }

        response = client.post("/api/v1/search", json=search_request)
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["score"] == 0.9

        # Verify search manager was called with correct parameters
        mock_search.assert_called_once()

    def test_search_endpoint_validation_error(self, client: TestClient):
        """Test search endpoint with invalid input."""
        # Missing required query field
        invalid_request = {"search_type": "hybrid", "max_results": 10}

        response = client.post("/api/v1/search", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_search_endpoint_empty_query(self, client: TestClient):
        """Test search endpoint with empty query."""
        search_request = {"query": "", "search_type": "hybrid", "max_results": 10}

        response = client.post("/api/v1/search", json=search_request)
        assert response.status_code == 422  # Should fail validation


@pytest.mark.unit
class TestDocumentEndpoints:
    """Test document management API endpoints."""

    @patch(
        "app.services.ingestion.document_processor.DocumentProcessor.process_document"
    )
    def test_upload_document(self, mock_process, client: TestClient):
        """Test document upload endpoint."""
        # Mock document processing
        mock_process.return_value = {
            "document_id": "doc123",
            "title": "Test Document",
            "chunks_created": 5,
            "processing_time": 2.3,
        }

        # Create a test file
        test_file_content = b"This is test document content for processing."
        files = {"file": ("test.txt", test_file_content, "text/plain")}

        response = client.post("/api/v1/documents/upload", files=files)
        assert response.status_code == 200

        data = response.json()
        assert data["document_id"] == "doc123"
        assert data["chunks_created"] == 5

        # Verify processor was called
        mock_process.assert_called_once()

    def test_upload_document_no_file(self, client: TestClient):
        """Test document upload without file."""
        response = client.post("/api/v1/documents/upload")
        assert response.status_code == 422  # Missing file

    @patch("app.models.database.get_session")
    def test_list_documents(self, mock_session, client: TestClient):
        """Test list documents endpoint."""
        # Mock database session and query results
        mock_db = AsyncMock()
        mock_session.return_value = mock_db

        with patch("app.api.v1.endpoints.documents.get_documents") as mock_get_docs:
            mock_get_docs.return_value = [
                {
                    "id": "doc1",
                    "title": "Document 1",
                    "created_at": "2024-01-01T00:00:00Z",
                    "file_type": "pdf",
                    "file_size": 1024,
                },
                {
                    "id": "doc2",
                    "title": "Document 2",
                    "created_at": "2024-01-02T00:00:00Z",
                    "file_type": "txt",
                    "file_size": 512,
                },
            ]

            response = client.get("/api/v1/documents")
            assert response.status_code == 200

            data = response.json()
            assert len(data) == 2
            assert data[0]["id"] == "doc1"
            assert data[1]["id"] == "doc2"

    @patch("app.models.database.get_session")
    def test_get_document_by_id(self, mock_session, client: TestClient):
        """Test get specific document endpoint."""
        mock_db = AsyncMock()
        mock_session.return_value = mock_db

        with patch("app.api.v1.endpoints.documents.get_document_by_id") as mock_get_doc:
            mock_get_doc.return_value = {
                "id": "doc1",
                "title": "Test Document",
                "content": "Document content...",
                "metadata": {"author": "Test Author"},
                "created_at": "2024-01-01T00:00:00Z",
            }

            response = client.get("/api/v1/documents/doc1")
            assert response.status_code == 200

            data = response.json()
            assert data["id"] == "doc1"
            assert data["title"] == "Test Document"
            assert "content" in data

    def test_get_document_not_found(self, client: TestClient):
        """Test get non-existent document."""
        with patch("app.api.v1.endpoints.documents.get_document_by_id") as mock_get_doc:
            mock_get_doc.return_value = None

            response = client.get("/api/v1/documents/nonexistent")
            assert response.status_code == 404

    @patch(
        "app.services.ingestion.document_processor.DocumentProcessor.delete_document"
    )
    def test_delete_document(self, mock_delete, client: TestClient):
        """Test document deletion endpoint."""
        mock_delete.return_value = True

        response = client.delete("/api/v1/documents/doc1")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Document deleted successfully"

        # Verify delete was called with correct ID
        mock_delete.assert_called_once_with("doc1")

    def test_delete_document_not_found(self, client: TestClient):
        """Test deleting non-existent document."""
        with patch(
            "app.services.ingestion.document_processor.DocumentProcessor."
            "delete_document"
        ) as mock_delete:
            mock_delete.return_value = False

            response = client.delete("/api/v1/documents/nonexistent")
            assert response.status_code == 404


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for API endpoints."""

    @pytest.mark.slow
    def test_full_document_workflow(self, client: TestClient):
        """Test complete document upload and search workflow."""
        with patch(
            "app.services.ingestion.document_processor.DocumentProcessor"
        ) as mock_processor_class:
            with patch(
                "app.services.search.search_manager.SearchManager"
            ) as mock_search_class:
                # Mock processor instance
                mock_processor = AsyncMock()
                mock_processor.process_document.return_value = {
                    "document_id": "test-doc",
                    "title": "Test Document",
                    "chunks_created": 3,
                }
                mock_processor_class.return_value = mock_processor

                # Mock search manager instance
                mock_search = AsyncMock()
                mock_search.search.return_value = {
                    "results": [
                        {
                            "id": "test-doc",
                            "score": 0.95,
                            "content": "Test content from uploaded document",
                            "metadata": {"title": "Test Document"},
                        }
                    ],
                    "total": 1,
                    "query_time": 0.1,
                }
                mock_search_class.return_value = mock_search

                # 1. Upload document
                test_content = b"This is a test document with machine learning content."
                files = {"file": ("test.txt", test_content, "text/plain")}

                upload_response = client.post("/api/v1/documents/upload", files=files)
                assert upload_response.status_code == 200

                upload_data = upload_response.json()
                document_id = upload_data["document_id"]

                # 2. Search for content
                search_request = {
                    "query": "machine learning",
                    "search_type": "hybrid",
                    "max_results": 5,
                }

                search_response = client.post("/api/v1/search", json=search_request)
                assert search_response.status_code == 200

                search_data = search_response.json()
                assert len(search_data["results"]) > 0

                # Verify the uploaded document appears in search results
                found_doc = next(
                    (r for r in search_data["results"] if r["id"] == document_id), None
                )
                assert found_doc is not None
                assert found_doc["score"] > 0.8


@pytest.mark.performance
class TestAPIPerformance:
    """Performance tests for API endpoints."""

    def test_search_response_time(self, client: TestClient, performance_threshold):
        """Test search endpoint response time."""
        import time

        with patch(
            "app.services.search.search_manager.SearchManager.search"
        ) as mock_search:
            mock_search.return_value = {"results": [], "total": 0, "query_time": 0.05}

            search_request = {
                "query": "test query",
                "search_type": "vector",
                "max_results": 10,
            }

            start_time = time.time()
            response = client.post("/api/v1/search", json=search_request)
            end_time = time.time()

            assert response.status_code == 200
            response_time = end_time - start_time
            assert response_time < performance_threshold["vector_search"]

    def test_concurrent_search_requests(self, client: TestClient):
        """Test handling of concurrent search requests."""
        import concurrent.futures

        with patch(
            "app.services.search.search_manager.SearchManager.search"
        ) as mock_search:
            mock_search.return_value = {"results": [], "total": 0, "query_time": 0.1}

            def make_search_request(query_id):
                search_request = {
                    "query": f"test query {query_id}",
                    "search_type": "hybrid",
                    "max_results": 10,
                }
                return client.post("/api/v1/search", json=search_request)

            # Make 10 concurrent requests
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                futures = [executor.submit(make_search_request, i) for i in range(10)]
                results = [future.result() for future in futures]

            # All requests should succeed
            assert all(r.status_code == 200 for r in results)

            # Search should have been called 10 times
            assert mock_search.call_count == 10


@pytest.mark.external
class TestExternalDependencies:
    """Tests that require external services (marked for optional execution)."""

    def test_health_check_with_real_dependencies(self, client: TestClient):
        """Test health check with real external dependencies."""
        # This test would run against actual Redis, DB, etc.
        # Only run when external services are available
        response = client.get("/api/v1/health/detailed")

        # Should return valid response structure regardless of service status
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "redis" in data

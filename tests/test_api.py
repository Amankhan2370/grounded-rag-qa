"""
API endpoint tests.
"""
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_query_endpoint():
    """Test query endpoint (requires vector DB setup)."""
    # This test would require a properly configured vector database
    # Skipping for now, but structure is here
    pass


def test_document_ingest_endpoint():
    """Test document ingestion endpoint."""
    # This test would require a sample document file
    # Skipping for now, but structure is here
    pass

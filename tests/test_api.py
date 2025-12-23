from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@patch('api.routes.RAGPipeline')
def test_query_endpoint(mock_rag):
    # Setup mock
    mock_instance = mock_rag.return_value
    mock_instance.query.return_value = {
        "answer": "Test answer",
        "sources": [],
        "context": "Test context",
        "num_sources": 0,
        "rewritten_query": None
    }
    
    response = client.post(
        "/api/v1/query",
        json={"question": "test question", "top_k": 2}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Test answer"
    assert "sources" in data

@patch('api.routes.ingest_documents')
@patch('api.routes.VectorDatabase')
def test_ingest_endpoint(mock_db, mock_ingest):
    # Setup mocks
    mock_ingest.return_value = ([MagicMock()], [MagicMock()])
    
    response = client.post(
        "/api/v1/ingest",
        json={"data_dir": "test_data"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["num_documents"] > 0

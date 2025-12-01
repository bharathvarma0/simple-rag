import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from generation.rag import RAGPipeline

@pytest.fixture
def mock_settings():
    with patch('generation.rag.get_settings') as mock:
        mock.return_value.vector_store.persist_dir = "test_store"
        mock.return_value.embedding.model_name = "test-model"
        mock.return_value.retrieval.top_k = 2
        mock.return_value.memory.enabled = False
        yield mock

@patch('generation.rag.VectorDatabase')
@patch('generation.rag.SimilaritySearch')
@patch('generation.rag.LLMWrapper')
def test_rag_query(mock_llm, mock_search, mock_db, mock_settings):
    # Setup mocks
    mock_db.return_value.exists.return_value = True
    
    mock_search.return_value.search.return_value = [
        {"metadata": {"source": "doc1", "text": "content1"}, "similarity_score": 0.9, "distance": 0.1},
        {"metadata": {"source": "doc2", "text": "content2"}, "similarity_score": 0.8, "distance": 0.2}
    ]
    mock_search.return_value.get_context.return_value = "content1\ncontent2"
    
    mock_llm.return_value.generate.return_value = "This is the answer."
    
    # Initialize pipeline
    rag = RAGPipeline()
    
    # Run query
    result = rag.query("test question")
    
    # Assertions
    assert result["answer"] == "This is the answer."
    assert len(result["sources"]) == 2
    assert result["sources"][0]["source"] == "doc1"
    assert result["context"] == "content1\ncontent2"
    
    # Verify calls
    mock_search.return_value.search.assert_called_once()
    mock_llm.return_value.generate.assert_called()

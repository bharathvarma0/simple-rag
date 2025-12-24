import sys
import os
from pathlib import Path
import shutil
import pytest
from unittest.mock import MagicMock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from augmentation.vector_db import VectorDatabase
from augmentation.search import SimilaritySearch
from generation.memory import ConversationMemory
from generation.rag import RAGPipeline
from config import get_settings

# Test Data
TEST_DOCS = [
    MagicMock(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "doc1", "page": 1}),
    MagicMock(page_content="Artificial Intelligence is transforming the world.", metadata={"source": "doc2", "page": 1}),
    MagicMock(page_content="Python is a popular programming language for AI.", metadata={"source": "doc3", "page": 1}),
    MagicMock(page_content="RAG systems combine retrieval and generation.", metadata={"source": "doc4", "page": 1}),
]

@pytest.fixture
def setup_teardown():
    # Setup
    settings = get_settings()
    original_persist_dir = settings.vector_store.persist_dir
    test_dir = "test_vector_store"
    settings.vector_store.persist_dir = test_dir
    
    yield
    
    # Teardown
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    settings.vector_store.persist_dir = original_persist_dir

def test_qdrant_integration(setup_teardown):
    print("\n[TEST] Verifying Qdrant Integration...")
    
    # 1. Initialize Vector DB
    vector_db = VectorDatabase()
    assert vector_db.collection_name == "rag_collection"
    
    # 2. Build from documents
    vector_db.build_from_documents(TEST_DOCS)
    
    # 3. Verify existence
    assert vector_db.exists()
    
    # 4. Verify search
    # Mock embedding for "AI"
    dummy_embedding = vector_db.embedder.generate_embedding("AI").astype('float32')
    dummy_embedding = dummy_embedding.reshape(1, -1)
    
    distances, indices, metadatas = vector_db.search(dummy_embedding, k=2)
    
    assert len(indices) > 0
    assert len(metadatas) > 0
    print("[PASS] Qdrant Integration Verified")

def test_retrieval_optimization(setup_teardown):
    print("\n[TEST] Verifying Retrieval Optimization...")
    
    # Setup DB
    vector_db = VectorDatabase()
    vector_db.build_from_documents(TEST_DOCS)
    
    # Initialize Search
    search = SimilaritySearch(vector_db)
    
    # 1. Test Search
    results = search.search("AI", top_k=2)
    assert len(results) > 0
    assert "similarity_score" in results[0]
    assert "metadata" in results[0]
    
    # 2. Test Cache
    # First call should cache
    search.search("cache test", top_k=1)
    assert "cache test" in search.cache
    
    # Second call should hit cache (mocking vector_db.search to fail if called)
    with patch.object(vector_db, 'search', side_effect=Exception("Should not be called")):
        cached_results = search.search("cache test", top_k=1)
        assert len(cached_results) > 0
        
    print("[PASS] Retrieval Optimization Verified")

def test_memory_summarization():
    print("\n[TEST] Verifying Memory Summarization...")
    
    memory = ConversationMemory()
    memory.window_size = 1 # Small window for testing
    
    # Add turns
    memory.add_turn("Hello", "Hi there")
    memory.add_turn("How are you?", "I'm good")
    memory.add_turn("What is AI?", "AI is...")
    
    # Mock LLM
    mock_llm = MagicMock()
    mock_llm.generate.return_value = "User asked about AI."
    
    # Summarize
    memory.summarize(mock_llm)
    
    assert memory.summary == "User asked about AI."
    assert len(memory.history) == 2 # Should keep last window_size * 2 turns (1 * 2 = 2)
    
    print("[PASS] Memory Summarization Verified")

if __name__ == "__main__":
    # Manual run if needed
    try:
        # We need to run pytest programmatically or just call functions
        # For simplicity in this script, we'll just call functions with a mock context
        # But pytest is better. Let's try to run pytest via command line.
        pass
    except Exception as e:
        print(f"Test failed: {e}")

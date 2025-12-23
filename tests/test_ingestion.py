import pytest
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from components.chunkers import DocumentChunker
from langchain_core.documents import Document

def test_document_chunker():
    # Test data
    text = "This is a test document.\n" * 100
    doc = Document(page_content=text, metadata={"source": "test.txt"})
    
    # Initialize chunker
    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    
    # Chunk documents
    chunks = chunker.chunk_documents([doc])
    
    # Assertions
    assert len(chunks) > 0
    assert isinstance(chunks[0], Document)
    assert len(chunks[0].page_content) <= 100
    assert chunks[0].metadata["source"] == "test.txt"

def test_chunker_empty_list():
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents([])
    assert len(chunks) == 0

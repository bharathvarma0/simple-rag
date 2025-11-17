"""
Text chunking utilities for splitting documents
"""

from typing import List, Any
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentChunker:
    """Split documents into smaller chunks for better embedding"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize document chunker
        
        Args:
            chunk_size: Maximum size of each chunk (in characters)
            chunk_overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        """
        Split documents into smaller chunks
        
        Args:
            documents: List of LangChain Document objects
            
        Returns:
            List of chunked Document objects
        """
        chunks = self.splitter.split_documents(documents)
        logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks


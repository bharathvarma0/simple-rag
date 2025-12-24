"""
Main data ingestion script
Loads documents, chunks them, and prepares for embedding
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Any, List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.loaders import DocumentLoader
from components.chunkers import DocumentChunker
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)


def ingest_documents(
    data_dir: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None
) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """
    Ingest documents from data directory
    
    Args:
        data_dir: Path to directory containing documents (uses config default if None)
        chunk_size: Size of text chunks (uses config default if None)
        chunk_overlap: Overlap between chunks (uses config default if None)
        
    Returns:
        Tuple of (raw_documents, chunked_documents)
    """
    settings = get_settings()
    data_dir = data_dir or settings.data.data_dir
    chunk_size = chunk_size or settings.chunking.chunk_size
    chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
    
    logger.info("=" * 60)
    logger.info("Starting Data Ingestion")
    logger.info("=" * 60)
    
    # Load documents using the new explicit loader
    from components.loaders import load_pdfs_from_dir
    raw_documents = load_pdfs_from_dir(data_dir)
    
    if not raw_documents:
        logger.error("No documents found or processed!")
        raise ValueError("No documents found or processed")
    
    # Chunk documents
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_documents = chunker.chunk_documents(raw_documents)
    
    logger.info("=" * 60)
    logger.info("Data Ingestion Complete")
    logger.info(f"Total raw documents: {len(raw_documents)}")
    logger.info(f"Total chunks: {len(chunked_documents)}")
    logger.info("=" * 60)
    
    return raw_documents, chunked_documents


if __name__ == "__main__":
    # Example usage
    data_directory = "data/pdfs"
    
    # Check if directory exists
    if not Path(data_directory).exists():
        print(f"[ERROR] Data directory '{data_directory}' not found!")
        print(f"[INFO] Please create the directory and add your PDF files")
        sys.exit(1)
    
    raw_docs, chunks = ingest_documents(data_directory)
    
    if chunks:
        print(f"\n[INFO] Sample chunk:")
        print(f"Content: {chunks[0].page_content[:200]}...")
        print(f"Metadata: {chunks[0].metadata}")


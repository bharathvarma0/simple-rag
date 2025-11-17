"""
Main application entry point
Traditional RAG system
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from data_ingestion.ingest import ingest_documents
from augmentation.vector_db import VectorDatabase
from generation.rag import RAGPipeline
from utils.logger import setup_logging, get_logger
from utils.formatters import format_results, format_sources
from config import get_settings

# Setup logging
setup_logging()
logger = get_logger(__name__)


def build_vector_store(data_dir: str = None, persist_dir: str = None):
    """
    Build vector store from documents
    
    Args:
        data_dir: Directory containing documents (uses config default if None)
        persist_dir: Directory to save vector store (uses config default if None)
    """
    settings = get_settings()
    data_dir = data_dir or settings.data.data_dir
    persist_dir = persist_dir or settings.vector_store.persist_dir
    
    logger.info("=" * 60)
    logger.info("Building Vector Store")
    logger.info("=" * 60)
    
    # Ingest documents
    raw_docs, chunks = ingest_documents(
        data_dir,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap
    )
    
    if not chunks:
        logger.error("No documents to process!")
        return
    
    # Build vector database
    vector_db = VectorDatabase(
        persist_dir=persist_dir,
        embedding_model=settings.embedding.model_name
    )
    vector_db.build_from_documents(
        raw_docs,
        chunk_size=settings.chunking.chunk_size,
        chunk_overlap=settings.chunking.chunk_overlap
    )
    
    logger.info("Vector store built successfully!")


def run_rag_query(question: str, top_k: int = None):
    """
    Run a RAG query
    
    Args:
        question: User question
        top_k: Number of documents to retrieve (uses config default if None)
    """
    settings = get_settings()
    top_k = top_k or settings.retrieval.top_k
    
    logger.info("=" * 60)
    logger.info("RAG Query")
    logger.info("=" * 60)
    
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Query
        result = rag.query(question, top_k=top_k)
        
        # Display results using formatter
        print(f"\nQuestion: {question}")
        print(format_results(result))
        
        return result
    
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        print(f"[ERROR] {e}")
        return None


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Traditional RAG System")
    parser.add_argument(
        "--build",
        action="store_true",
        help="Build vector store from documents"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Question to ask the RAG system"
    )
    settings = get_settings()
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=settings.data.data_dir,
        help=f"Directory containing documents (default: {settings.data.data_dir})"
    )
    parser.add_argument(
        "--vector-store",
        type=str,
        default=settings.vector_store.persist_dir,
        help=f"Directory for vector store (default: {settings.vector_store.persist_dir})"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.retrieval.top_k,
        help=f"Number of documents to retrieve (default: {settings.retrieval.top_k})"
    )
    
    args = parser.parse_args()
    
    if args.build:
        build_vector_store(args.data_dir, args.vector_store)
    elif args.query:
        run_rag_query(args.query, args.top_k)
    else:
        # Interactive mode
        print("Traditional RAG System")
        print("=" * 60)
        
        # Check if vector store exists
        vector_store_path = Path(args.vector_store)
        if not vector_store_path.exists():
            print("[INFO] Vector store not found. Building...")
            build_vector_store(args.data_dir, args.vector_store)
        
        # Interactive query loop
        print("\nEnter questions (type 'exit' to quit):")
        while True:
            question = input("\n> ")
            if question.lower() in ['exit', 'quit', 'q']:
                break
            
            if question.strip():
                run_rag_query(question, args.top_k)


if __name__ == "__main__":
    main()

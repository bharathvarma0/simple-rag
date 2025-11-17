"""
Vector database management using FAISS
"""

import faiss
import numpy as np
import pickle
from typing import List, Any, Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.embedders import EmbeddingGenerator
from components.chunkers import DocumentChunker
from utils.logger import get_logger
from utils.paths import ensure_dir
from config import get_settings

logger = get_logger(__name__)


class VectorDatabase:
    """Manage vector database using FAISS"""
    
    def __init__(self, persist_dir: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initialize vector database
        
        Args:
            persist_dir: Directory to persist the vector store (uses config default if None)
            embedding_model: Embedding model name (uses config default if None)
        """
        settings = get_settings()
        persist_dir = persist_dir or settings.vector_store.persist_dir
        embedding_model = embedding_model or settings.embedding.model_name
        
        self.persist_dir = ensure_dir(persist_dir)
        
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        
        logger.info(f"Vector database initialized at: {self.persist_dir}")
    
    def build_from_documents(
        self, 
        documents: List[Any], 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ):
        """
        Build vector database from documents
        
        Args:
            documents: List of LangChain Document objects
            chunk_size: Size of text chunks (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)
        """
        settings = get_settings()
        chunk_size = chunk_size or settings.chunking.chunk_size
        chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        
        logger.info(f"Building vector database from {len(documents)} documents...")
        
        # Chunk documents
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(documents)
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(texts)
        
        # Store metadata
        metadatas = [{"text": chunk.page_content, **chunk.metadata} for chunk in chunks]
        
        # Add to index
        self.add_embeddings(embeddings.astype('float32'), metadatas)
        
        # Save
        self.save()
        logger.info("Vector database built and saved")
    
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict] = None):
        """
        Add embeddings to the vector database
        
        Args:
            embeddings: NumPy array of embeddings
            metadatas: List of metadata dictionaries
        """
        dim = embeddings.shape[1]
        
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        
        self.index.add(embeddings)
        
        if metadatas:
            self.metadata.extend(metadatas)
        
        logger.info(f"Added {embeddings.shape[0]} vectors to database")
    
    def save(self):
        """Save vector database to disk"""
        faiss_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"
        
        if self.index is None:
            logger.warning("No index to save")
            return
        
        faiss.write_index(self.index, str(faiss_path))
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        
        logger.info(f"Saved vector database to {self.persist_dir}")
    
    def load(self):
        """Load vector database from disk"""
        faiss_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"
        
        if not (faiss_path.exists() and meta_path.exists()):
            raise FileNotFoundError(f"Vector database not found at {self.persist_dir}")
        
        self.index = faiss.read_index(str(faiss_path))
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        logger.info(f"Loaded vector database from {self.persist_dir}")
        logger.info(f"Total vectors in database: {self.index.ntotal}")
    
    def exists(self) -> bool:
        """Check if vector database exists"""
        faiss_path = self.persist_dir / "faiss.index"
        meta_path = self.persist_dir / "metadata.pkl"
        return faiss_path.exists() and meta_path.exists()


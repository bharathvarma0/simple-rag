"""
Similarity search utilities for vector database
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from augmentation.vector_db import VectorDatabase
from components.embedders import EmbeddingGenerator
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)


class SimilaritySearch:
    """Handle similarity search in vector database"""
    
    def __init__(self, vector_db: VectorDatabase):
        """
        Initialize similarity search
        
        Args:
            vector_db: VectorDatabase instance
        """
        self.vector_db = vector_db
        self.embedder = EmbeddingGenerator(model_name=vector_db.embedding_model)
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query text
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List of search results with metadata and distances
        """
        if self.vector_db.index is None:
            raise ValueError("Vector database not loaded. Call load() first.")
        
        settings = get_settings()
        top_k = top_k or settings.retrieval.top_k
        
        logger.info(f"Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.generate_embedding(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search in FAISS
        distances, indices = self.vector_db.index.search(query_embedding, top_k)
        
        # Format results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.vector_db.metadata):
                result = {
                    "index": int(idx),
                    "distance": float(dist),
                    "similarity_score": float(1 / (1 + dist)),  # Convert distance to similarity
                    "metadata": self.vector_db.metadata[idx]
                }
                results.append(result)
        
        logger.info(f"Found {len(results)} results")
        return results
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get context text from search results
        
        Args:
            query: Search query text
            top_k: Number of top results to use (uses config default if None)
            
        Returns:
            Combined context text from search results
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        # Extract text from metadata
        texts = [r["metadata"].get("text", "") for r in results]
        context = "\n\n".join(texts)
        
        return context


"""
Embedding generation utilities
"""

from typing import List
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text documents"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        logger.info(f"Loaded embedding model: {model_name}")
        logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Single text string to embed
            
        Returns:
            NumPy array of embedding with shape (embedding_dim,)
        """
        embedding = self.model.encode([text])
        return embedding[0]


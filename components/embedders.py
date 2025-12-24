"""
Embedding generation utilities
"""

from typing import List
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from langchain_openai import OpenAIEmbeddings
from utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """Generate embeddings for text documents"""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize embedding generator
        
        Args:
            model_name: OpenAI model name for embeddings
        """
        self.model_name = model_name
        self.model = OpenAIEmbeddings(model=model_name)
        logger.info(f"Loaded embedding model: {model_name}")
        # logger.info(f"Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.embed_documents(texts)
        embeddings = np.array(embeddings)
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
        embedding = self.model.embed_query(text)
        return np.array(embedding)


"""
Local embedding provider using sentence-transformers
"""

from providers.base_provider import BaseEmbeddingProvider
from typing import List
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """Local embedding provider using sentence-transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self._model = None
        self._dimension = None
        logger.info(f"Initialized local embedding provider with model: {model_name}")
    
    def _get_model(self):
        """Lazy load embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Get dimension from a test embedding
                test_emb = self._model.encode("test")
                self._dimension = len(test_emb)
                logger.info(f"Loaded embedding model, dimension: {self._dimension}")
            except ImportError:
                logger.error("sentence-transformers not installed")
                raise ImportError("sentence-transformers not installed. Run: pip install sentence-transformers")
        return self._model
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text"""
        model = self._get_model()
        embedding = model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch of texts"""
        model = self._get_model()
        embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return embeddings
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        if self._dimension is None:
            self._get_model()  # This will set the dimension
        return self._dimension

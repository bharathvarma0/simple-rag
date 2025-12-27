"""
Re-Ranking Component for RAG System

Provides cross-encoder based re-ranking of retrieved chunks for better accuracy.
Configurable via strategies.yaml.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract base class for reranking implementations"""
    
    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks based on relevance to query
        
        Args:
            query: The search query
            chunks: List of chunks with 'content' and 'metadata'
            top_k: Number of top chunks to return
            
        Returns:
            Re-ranked list of top_k chunks
        """
        pass


class CrossEncoderReranker(BaseReranker):
    """Cross-encoder based reranking using sentence-transformers"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize cross-encoder reranker
        
        Args:
            config: Configuration dict with:
                - model: Model name (default: cross-encoder/ms-marco-MiniLM-L-6-v2)
                - batch_size: Batch size for inference (default: 32)
                - device: Device to use (default: cpu)
        """
        self.config = config or {}
        self.model_name = self.config.get('model', 'cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.batch_size = self.config.get('batch_size', 32)
        self.device = self.config.get('device', 'cpu')
        
        self._model = None  # Lazy loading
        
        logger.info(f"Initialized CrossEncoderReranker with model: {self.model_name}")
    
    def _load_model(self):
        """Lazy load the cross-encoder model"""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
                
                logger.info(f"Loading cross-encoder model: {self.model_name}")
                self._model = CrossEncoder(self.model_name, device=self.device)
                logger.info(f"Model loaded successfully on device: {self.device}")
                
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise
        
        return self._model
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Re-rank chunks using cross-encoder
        
        Args:
            query: The search query
            chunks: List of chunks (dicts with 'page_content' or 'content' field)
            top_k: Number of top chunks to return
        
        Returns:
            Re-ranked list of top_k chunks with added 'rerank_score'
        """
        if not chunks:
            return []
        
        # Limit top_k to available chunks
        top_k = min(top_k, len(chunks))
        
        # Load model if not loaded
        model = self._load_model()
        
        # Extract text content from chunks
        texts = []
        for chunk in chunks:
            # Handle different chunk formats
            if hasattr(chunk, 'page_content'):
                texts.append(chunk.page_content)
            elif isinstance(chunk, dict):
                texts.append(chunk.get('content') or chunk.get('page_content', ''))
            else:
                texts.append(str(chunk))
        
        # Create query-text pairs
        pairs = [[query, text] for text in texts]
        
        # Score all pairs
        logger.debug(f"Re-ranking {len(pairs)} chunks with query: {query[:100]}...")
        scores = model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        
        # Combine chunks with scores
        scored_chunks = []
        for chunk, score in zip(chunks, scores):
            # Add rerank score to chunk
            if isinstance(chunk, dict):
                chunk_copy = chunk.copy()
                chunk_copy['rerank_score'] = float(score)
                scored_chunks.append(chunk_copy)
            else:
                # For Langchain Document objects
                chunk_copy = chunk
                if hasattr(chunk_copy, 'metadata'):
                    chunk_copy.metadata['rerank_score'] = float(score)
                else:
                    setattr(chunk_copy, 'rerank_score', float(score))
                scored_chunks.append(chunk_copy)
        
        # Sort by score (descending)
        scored_chunks.sort(
            key=lambda x: x.get('rerank_score') if isinstance(x, dict) 
                         else getattr(x, 'rerank_score', x.metadata.get('rerank_score', 0)),
            reverse=True
        )
        
        # Return top_k
        top_chunks = scored_chunks[:top_k]
        
        logger.info(f"Re-ranked {len(chunks)} chunks → top {len(top_chunks)}")
        if top_chunks:
            top_score = (top_chunks[0].get('rerank_score') if isinstance(top_chunks[0], dict)
                        else getattr(top_chunks[0], 'rerank_score', 
                                   top_chunks[0].metadata.get('rerank_score', 0)))
            logger.debug(f"Top chunk score: {top_score:.4f}")
        
        return top_chunks


class NoOpReranker(BaseReranker):
    """No-op reranker that returns chunks as-is (for when reranking is disabled)"""
    
    def rerank(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Return chunks as-is, limited to top_k"""
        return chunks[:top_k]


def create_reranker(config: Dict[str, Any] = None) -> BaseReranker:
    """
    Factory function to create reranker based on configuration
    
    Args:
        config: Configuration dict with 'enabled' and other params
        
    Returns:
        BaseReranker instance
    """
    if not config or not config.get('enabled', False):
        logger.info("Reranking disabled, using NoOpReranker")
        return NoOpReranker()
    
    reranker_type = config.get('type', 'cross_encoder')
    
    if reranker_type == 'cross_encoder':
        return CrossEncoderReranker(config)
    else:
        logger.warning(f"Unknown reranker type: {reranker_type}, using NoOp")
        return NoOpReranker()

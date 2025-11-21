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
        Search for similar documents using Hybrid Search (Vector + Keyword)
        
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
        
        # 1. Vector Search (Semantic)
        query_embedding = self.embedder.generate_embedding(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.vector_db.index.search(query_embedding, top_k * 2) # Get more candidates
        
        vector_results = {}
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.vector_db.metadata):
                # Normalize score (1 / (1 + distance))
                score = 1 / (1 + dist)
                vector_results[int(idx)] = {"score": score, "rank": rank}

        # 2. Keyword Search (BM25)
        keyword_results = {}
        # Lazy load keyword db if needed
        if not hasattr(self, 'keyword_db'):
            from augmentation.keyword_db import KeywordDatabase
            self.keyword_db = KeywordDatabase(self.vector_db.persist_dir)
            if self.keyword_db.exists():
                self.keyword_db.load()
        
        if hasattr(self, 'keyword_db') and self.keyword_db.bm25:
            bm25_res = self.keyword_db.search(query, top_k=top_k * 2)
            for rank, res in enumerate(bm25_res):
                # Normalize BM25 score (simple min-max normalization would be better, but this is a rough approx)
                # For now, we use the raw score but rely on RRF for ranking
                keyword_results[res["id"]] = {"score": res["score"], "rank": rank}
        
        # 3. Reciprocal Rank Fusion (RRF)
        # score = 1 / (k + rank)
        k = 60
        combined_scores = {}
        
        # Process Vector Results
        for idx, data in vector_results.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (k + data["rank"]))
            
        # Process Keyword Results
        for idx, data in keyword_results.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (k + data["rank"]))
            
        # Sort by combined score
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
        
        # Format final results
        results = []
        for idx in sorted_indices:
            if idx < len(self.vector_db.metadata):
                # We use the vector distance as the primary "distance" metric for compatibility
                # But the ranking is determined by RRF
                original_dist = 0.0
                if idx in vector_results:
                    original_dist = (1 / vector_results[idx]["score"]) - 1
                
                result = {
                    "index": int(idx),
                    "distance": float(original_dist),
                    "similarity_score": float(combined_scores[idx]), # RRF Score
                    "metadata": self.vector_db.metadata[idx],
                    "sources": []
                }
                
                if idx in vector_results: result["sources"].append("vector")
                if idx in keyword_results: result["sources"].append("keyword")
                
                results.append(result)
        
        logger.info(f"Found {len(results)} hybrid results")
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


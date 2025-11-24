"""
Similarity search utilities for vector database with multi-stage retrieval
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
    """Handle similarity search in vector database with multi-stage retrieval support"""
    
    def __init__(self, vector_db: VectorDatabase):
        """
        Initialize similarity search
        
        Args:
            vector_db: VectorDatabase instance
        """
        self.vector_db = vector_db
        self.embedder = EmbeddingGenerator(model_name=vector_db.embedding_model)
        self._chunk_context = None  # Lazy load
    
    @property
    def chunk_context(self):
        """Lazy load chunk context manager"""
        if self._chunk_context is None:
            try:
                from augmentation.chunk_context import ChunkContext
                self._chunk_context = ChunkContext()
                logger.info("Initialized chunk context manager")
            except Exception as e:
                logger.warning(f"Could not initialize chunk context: {e}")
                self._chunk_context = None
        return self._chunk_context
    
    def search(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        retrieval_depth: int = 1,
        initial_candidates: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Multi-stage hybrid search with optional context expansion
        
        Args:
            query: Search query text
            top_k: Number of final results to return
            retrieval_depth: Stage depth (1=basic, 2=with neighbors)
            initial_candidates: Initial retrieval count for stage 1
            
        Returns:
            List of search results with metadata and distances
        """
        if self.vector_db.index is None:
            raise ValueError("Vector database not loaded. Call load() first.")
        
        settings = get_settings()
        top_k = top_k or settings.retrieval.top_k
        
        # Determine initial retrieval count
        if retrieval_depth > 1 and initial_candidates:
            search_k = initial_candidates
        else:
            search_k = top_k
        
        logger.info(f"Searching for: '{query}'")
        
        # Stage 1: Hybrid Search (Vector + Keyword)
        results = self._hybrid_search(query, search_k)
        
        # Stage 2: Context Expansion (if depth > 1)
        if retrieval_depth > 1 and self.chunk_context and len(results) > 0:
            logger.info(f"Expanding {len(results)} chunks with neighbors (depth={retrieval_depth})")
            expanded = self._expand_with_context(results, depth=retrieval_depth)
            results = expanded
        
        # Return final top_k
        return results[:top_k]
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Core hybrid search logic (Vector + Keyword with RRF)
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            Search results
        """
        # 1. Vector Search (Semantic)
        query_embedding = self.embedder.generate_embedding(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        distances, indices = self.vector_db.index.search(query_embedding, top_k * 2)
        
        vector_results = {}
        for rank, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.vector_db.metadata):
                score = 1 / (1 + dist)
                vector_results[int(idx)] = {"score": score, "rank": rank}

        # 2. Keyword Search (BM25)
        keyword_results = {}
        if not hasattr(self, 'keyword_db'):
            from augmentation.keyword_db import KeywordDatabase
            self.keyword_db = KeywordDatabase(self.vector_db.persist_dir)
            if self.keyword_db.exists():
                self.keyword_db.load()
        
        if hasattr(self, 'keyword_db') and self.keyword_db.bm25:
            bm25_res = self.keyword_db.search(query, top_k=top_k * 2)
            for rank, res in enumerate(bm25_res):
                keyword_results[res["id"]] = {"score": res["score"], "rank": rank}
        
        # 3. Reciprocal Rank Fusion (RRF)
        k = 60
        combined_scores = {}
        
        for idx, data in vector_results.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (k + data["rank"]))
            
        for idx, data in keyword_results.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (k + data["rank"]))
            
        # Sort by combined score
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:top_k]
        
        # Format results
        results = []
        for idx in sorted_indices:
            if idx < len(self.vector_db.metadata):
                original_dist = 0.0
                if idx in vector_results:
                    original_dist = (1 / vector_results[idx]["score"]) - 1
                
                result = {
                    "index": int(idx),
                    "distance": float(original_dist),
                    "similarity_score": float(combined_scores[idx]),
                    "metadata": self.vector_db.metadata[idx],
                    "sources": [],
                    "page_content": self.vector_db.metadata[idx].get("text", "")  # For chunk context
                }
                
                if idx in vector_results: result["sources"].append("vector")
                if idx in keyword_results: result["sources"].append("keyword")
                
                results.append(result)
        
        logger.info(f"Found {len(results)} hybrid results")
        return results
    
    def _expand_with_context(
        self, 
        results: List[Dict[str, Any]], 
        depth: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Expand results with neighboring chunks
        
        Args:
            results: Initial search results
            depth: Neighbor depth
            
        Returns:
            Expanded results
        """
        try:
            # Convert results to format chunk_context expects
            chunks_collection = []
            for idx, meta in enumerate(self.vector_db.metadata):
                chunks_collection.append({
                    "index": idx,
                    "page_content": meta.get("text", ""),
                    "content": meta.get("text", ""),
                    "metadata": meta
                })
            
            # Expand
            expanded = self.chunk_context.expand_chunks(results, chunks_collection, depth)
            
            logger.info(f"Expanded to {len(expanded)} chunks")
            return expanded
            
        except Exception as e:
            logger.warning(f"Context expansion failed: {e}, returning original results")
            return results
    
    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get context text from search results
        
        Args:
            query: Search query text
            top_k: Number of top results to use
            
        Returns:
            Combined context text
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        texts = [r["metadata"].get("text", "") for r in results]
        context = "\n\n".join(texts)
        
        return context

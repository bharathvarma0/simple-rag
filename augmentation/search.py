"""
Similarity search utilities for vector database
"""

import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from augmentation.vector_db import VectorDatabase
from components.embedders import EmbeddingGenerator
from utils.logger import get_logger
from config import get_settings

try:
    from sentence_transformers import CrossEncoder
except ImportError:
    CrossEncoder = None

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
        self.settings = get_settings()
        
        # Retrieval Cache: query -> results
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        
        # Reranker
        self.reranker = None
        if self.settings.reranker.enabled:
            if CrossEncoder:
                try:
                    self.reranker = CrossEncoder(self.settings.reranker.model_name)
                    logger.info(f"Reranker initialized: {self.settings.reranker.model_name}")
                except Exception as e:
                    logger.warning(f"Failed to initialize reranker: {e}")
            else:
                logger.warning("sentence-transformers not installed, reranking disabled")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using Optimized Hybrid Search
        1. Check Cache
        2. BM25 Search (Top 50) -> Candidate Filtering
        3. Vector Search (Filtered)
        4. RRF Fusion
        5. Reranking (Optional)
        
        Args:
            query: Search query text
            top_k: Number of top results to return (uses config default if None)
            
        Returns:
            List of search results with metadata and distances
        """
        top_k = top_k or self.settings.retrieval.top_k
        
        # 1. Check Cache
        normalized_query = query.strip().lower()
        if normalized_query in self.cache:
            logger.info("Retrieval cache hit")
            return self.cache[normalized_query][:top_k]
        
        logger.info(f"Searching for: '{query}'")
        
        # Lazy load keyword db if needed - REMOVED
        # if not hasattr(self, 'keyword_db'):
        #     from augmentation.keyword_db import KeywordDatabase
        #     self.keyword_db = KeywordDatabase(self.vector_db.persist_dir)
        #     if self.keyword_db.exists():
        #         self.keyword_db.load()
        
        # 2. BM25 Search (Candidate Generation) - REMOVED
        bm25_candidates = []
        bm25_ids = []
        keyword_results = {}
        
        # if hasattr(self, 'keyword_db') and self.keyword_db.bm25:
        #     # Get more candidates for filtering (e.g., 50)
        #     candidate_k = 50
        #     bm25_res = self.keyword_db.search(query, top_k=candidate_k)
        #     
        #     for rank, res in enumerate(bm25_res):
        #         bm25_candidates.append(res)
        #         bm25_ids.append(res["id"])
        #         keyword_results[res["id"]] = {"score": res["score"], "rank": rank}
        #         
        #     logger.info(f"BM25 found {len(bm25_candidates)} candidates")
        
        # 3. Vector Search (Filtered)
        query_embedding = self.embedder.generate_embedding(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        
        # If we have candidates, filter vector search. Else, full search.
        filter_ids = bm25_ids if bm25_ids else None
        
        # We need enough vector results to fuse
        vector_k = top_k * 2 if not filter_ids else len(filter_ids)
        
        distances, indices, metadatas = self.vector_db.search(
            query_embedding, 
            k=vector_k, 
            filter_ids=filter_ids
        )
        
        vector_results = {}
        for rank, (idx, dist, meta) in enumerate(zip(indices, distances, metadatas)):
            if idx != -1:
                # Normalize score (1 / (1 + distance))
                score = 1 / (1 + dist)
                vector_results[int(idx)] = {
                    "score": score, 
                    "rank": rank,
                    "metadata": meta
                }

        # 4. Reciprocal Rank Fusion (RRF)
        k_rrf = 60
        combined_scores = {}
        all_metadata = {}
        
        # Process Vector Results
        for idx, data in vector_results.items():
            combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (k_rrf + data["rank"]))
            all_metadata[idx] = data["metadata"]
            
        # Process Keyword Results - REMOVED
        # for idx, data in keyword_results.items():
        #     combined_scores[idx] = combined_scores.get(idx, 0) + (1 / (k_rrf + data["rank"]))
        #     # If metadata not in vector results (rare if filtered, but possible if vector search missed it), add it
        #     if idx not in all_metadata and idx < len(self.keyword_db.metadata):
        #          all_metadata[idx] = self.keyword_db.metadata[idx]
        #          # Ensure text is in metadata
        #          if "text" not in all_metadata[idx]:
        #              all_metadata[idx]["text"] = self.keyword_db.documents[idx]
            
        # Sort by combined score
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        # Format initial results
        results = []
        for idx in sorted_indices:
            if idx in all_metadata:
                result = {
                    "index": int(idx),
                    "similarity_score": float(combined_scores[idx]), # RRF Score
                    "metadata": all_metadata[idx],
                    "sources": []
                }
                
                if idx in vector_results: result["sources"].append("vector")
                # if idx in keyword_results: result["sources"].append("keyword")
                
                results.append(result)
        
        # 5. Reranking (Optional)
        if self.reranker and results:
            results = self._rerank_results(query, results)
        
        # Limit to top_k
        final_results = results[:top_k]
        
        # Update Cache
        self.cache[normalized_query] = final_results
        
        logger.info(f"Found {len(final_results)} hybrid results")
        return final_results
    
    def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using Cross-Encoder
        """
        if not results:
            return results
            
        # Prepare pairs for reranking
        # We only rerank top 20 to save time
        rerank_limit = 20
        candidates = results[:rerank_limit]
        
        pairs = []
        for res in candidates:
            text = res["metadata"].get("text", "")
            pairs.append([query, text])
            
        # Predict scores
        scores = self.reranker.predict(pairs)
        
        # Update scores and resort
        for i, score in enumerate(scores):
            candidates[i]["similarity_score"] = float(score)
            candidates[i]["sources"].append("reranker")
            
        # Sort by new score
        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        # Combine with remaining results (if any)
        final_results = candidates + results[rerank_limit:]
        
        return final_results
    
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


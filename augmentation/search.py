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
from augmentation.keyword_db import KeywordDatabase
from components.embedders import EmbeddingGenerator
from components.reranker import create_reranker
from utils.logger import get_logger
from config import get_settings

from utils.cache import cache_result

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
        
        # Initialize Keyword DB
        self.keyword_db = KeywordDatabase(self.vector_db.persist_dir)
        if self.keyword_db.exists():
            self.keyword_db.load()
        else:
            logger.warning("Keyword database not found. Hybrid search will be limited to vector only.")
        
        # Initialize Reranker
        self.reranker = create_reranker({
            'enabled': self.settings.reranker.enabled,
            'model': self.settings.reranker.model_name,
            'device': 'cpu' # Auto-detect in production
        })
        
        # Retrieval Cache: query -> results
        self.cache: Dict[str, List[Dict[str, Any]]] = {}
        
    @cache_result(prefix="search", ttl=3600)
    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using Optimized Hybrid Search
        1. Check Cache
        2. BM25 Search (Top 50)
        3. Vector Search (Top 50)
        4. RRF Fusion
        5. Reranking
        
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
        
        # 2. BM25 Search
        bm25_results = []
        if self.keyword_db.bm25:
            # Get more candidates for fusion
            candidate_k = max(50, top_k * 2)
            bm25_results = self.keyword_db.search(query, top_k=candidate_k)
            logger.info(f"BM25 found {len(bm25_results)} candidates")
        
        # 3. Vector Search
        query_embedding = self.embedder.generate_embedding(query).astype('float32')
        query_embedding = query_embedding.reshape(1, -1)
        
        # Get more candidates for fusion
        vector_k = max(50, top_k * 2)
        distances, indices, metadatas = self.vector_db.search(
            query_embedding, 
            k=vector_k
        )
        
        vector_results = []
        for rank, (idx, dist, meta) in enumerate(zip(indices, distances, metadatas)):
            if idx != -1:
                # Normalize score (1 / (1 + distance)) for display, but RRF uses rank
                score = 1 / (1 + dist)
                vector_results.append({
                    "id": int(idx),
                    "score": score,
                    "metadata": meta,
                    "rank": rank
                })
        
        # 4. Reciprocal Rank Fusion (RRF)
        fused_results = self._rrf_fusion(bm25_results, vector_results, k=60)
        
        # 5. Reranking
        # Take top N for reranking
        rerank_limit = 20
        candidates = fused_results[:rerank_limit]
        
        if candidates:
            reranked = self.reranker.rerank(query, candidates, top_k=top_k)
            # Combine with remaining results if any (though usually we just return top_k)
            final_results = reranked
        else:
            final_results = fused_results[:top_k]
            
        # Update Cache
        self.cache[normalized_query] = final_results
        
        logger.info(f"Found {len(final_results)} hybrid results")
        return final_results

    def _rrf_fusion(self, 
                   keyword_results: List[Dict], 
                   vector_results: List[Dict], 
                   k: int = 60) -> List[Dict]:
        """
        Perform Reciprocal Rank Fusion
        """
        combined_scores = {}
        all_metadata = {}
        
        # Process Keyword Results
        for rank, res in enumerate(keyword_results):
            doc_id = res["id"]
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (k + rank))
            all_metadata[doc_id] = res["metadata"]
            all_metadata[doc_id]["text"] = res["text"] # Ensure text is present
            
        # Process Vector Results
        for res in vector_results:
            doc_id = res["id"]
            # Vector results rank is already in the dict
            rank = res["rank"]
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + (1 / (k + rank))
            if doc_id not in all_metadata:
                all_metadata[doc_id] = res["metadata"]
                
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        results = []
        for doc_id in sorted_ids:
            score = combined_scores[doc_id]
            meta = all_metadata[doc_id]
            
            # Determine sources
            sources = []
            if any(r["id"] == doc_id for r in keyword_results):
                sources.append("keyword")
            if any(r["id"] == doc_id for r in vector_results):
                sources.append("vector")
                
            results.append({
                "index": doc_id,
                "similarity_score": float(score),
                "metadata": meta,
                "sources": sources,
                "content": meta.get("text", "") # Standardize content field
            })
            
        return results

    def get_context(self, query: str, top_k: Optional[int] = None) -> str:
        """
        Get context text from search results
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return ""
        
        # Extract text from metadata
        texts = [r["metadata"].get("text", "") for r in results]
        context = "\n\n".join(texts)
        
        return context

"""
Keyword database management using BM25
"""

import pickle
from typing import List, Any, Dict, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rank_bm25 import BM25Okapi
from utils.logger import get_logger
from utils.paths import ensure_dir
from config import get_settings

logger = get_logger(__name__)


class KeywordDatabase:
    """Manage keyword database using BM25"""
    
    def __init__(self, persist_dir: Optional[str] = None):
        """
        Initialize keyword database
        
        Args:
            persist_dir: Directory to persist the keyword store
        """
        settings = get_settings()
        persist_dir = persist_dir or settings.vector_store.persist_dir
        
        self.persist_dir = ensure_dir(persist_dir)
        self.bm25 = None
        self.documents = []  # Store raw text for retrieval
        self.metadata = []
        
        logger.info(f"Keyword database initialized at: {self.persist_dir}")
    
    def build_from_documents(self, documents: List[Any]):
        """
        Build keyword database from documents
        
        Args:
            documents: List of LangChain Document objects or chunks
        """
        logger.info(f"Building keyword database from {len(documents)} documents...")
        
        # Tokenize corpus
        tokenized_corpus = [doc.page_content.lower().split() for doc in documents]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # Store documents and metadata
        self.documents = [doc.page_content for doc in documents]
        self.metadata = [doc.metadata for doc in documents]
        
        # Save
        self.save()
        logger.info("Keyword database built and saved")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for documents using keywords
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of results with scores
        """
        if self.bm25 is None:
            logger.warning("BM25 index not built")
            return []
            
        tokenized_query = query.lower().split()
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_n:
            if scores[idx] > 0:  # Only return relevant results
                results.append({
                    "text": self.documents[idx],
                    "metadata": self.metadata[idx],
                    "score": scores[idx],
                    "id": idx
                })
                
        return results
    
    def save(self):
        """Save keyword database to disk"""
        bm25_path = self.persist_dir / "bm25.pkl"
        data_path = self.persist_dir / "bm25_data.pkl"
        
        if self.bm25 is None:
            logger.warning("No index to save")
            return
        
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25, f)
            
        with open(data_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)
        
        logger.info(f"Saved keyword database to {self.persist_dir}")
    
    def load(self):
        """Load keyword database from disk"""
        bm25_path = self.persist_dir / "bm25.pkl"
        data_path = self.persist_dir / "bm25_data.pkl"
        
        if not (bm25_path.exists() and data_path.exists()):
            # It's okay if it doesn't exist yet, just log it
            logger.warning(f"Keyword database not found at {self.persist_dir}")
            return
        
        try:
            with open(bm25_path, "rb") as f:
                self.bm25 = pickle.load(f)
                
            with open(data_path, "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.metadata = data["metadata"]
                
            logger.info(f"Loaded keyword database from {self.persist_dir}")
        except Exception as e:
            logger.error(f"Failed to load keyword database: {e}")
            self.bm25 = None
            self.documents = []
            self.metadata = []
    
    def exists(self) -> bool:
        """Check if keyword database exists"""
        bm25_path = self.persist_dir / "bm25.pkl"
        return bm25_path.exists()

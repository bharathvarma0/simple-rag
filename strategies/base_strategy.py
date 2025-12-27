"""
Base strategy class with common functionality
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import sys
from pathlib import Path
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from core.base import BaseStrategy as CoreBaseStrategy, QueryProfile, DocumentProfile, StrategyParams
from augmentation.vector_db import VectorDatabase
from augmentation.search import SimilaritySearch
from providers.base_provider import BaseLLMProvider
from components.reranker import create_reranker
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseStrategy(CoreBaseStrategy):
    """Base strategy with common RAG functionality"""
    
    # Class variable to cache loaded config
    _config_cache = None
    _reranker_cache = None
    
    def __init__(self):
        self.vector_db = None
        self.search = None
        self.llm = None
        self.strategy_name = None  # Set by subclasses
        self.reranker = None
    
    @classmethod
    def _load_strategy_config(cls) -> Dict:
        """Load strategy configuration from YAML (cached)"""
        if cls._config_cache is not None:
            return cls._config_cache
        
        config_path = Path(__file__).parent.parent / "config" / "strategies.yaml"
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    cls._config_cache = yaml.safe_load(f)
                logger.info("Loaded strategy configuration from strategies.yaml")
                return cls._config_cache
            except Exception as e:
                logger.warning(f"Failed to load strategies.yaml: {e}, using hardcoded defaults")
                return {}
        else:
            logger.info("No strategies.yaml found, using hardcoded defaults")
            return {}
    
    def _get_config_params(self, strategy_name: str) -> Optional[Dict]:
        """Get parameters from YAML config for this strategy"""
        config = self._load_strategy_config()
        strategies = config.get('strategies', {})
        return strategies.get(strategy_name)
    
    def initialize(self, vector_db: VectorDatabase, llm: BaseLLMProvider):
        """Initialize strategy with vector DB and LLM"""
        self.vector_db = vector_db
        self.search = SimilaritySearch(vector_db)
        self.llm = llm
        
        # Initialize reranker (lazy, only if enabled)
        if self.__class__._reranker_cache is None:
            config = self._load_strategy_config()
            reranking_config = config.get('reranking', {})
            self.__class__._reranker_cache = create_reranker(reranking_config)
            logger.info(f"Initialized reranker: {type(self.__class__._reranker_cache).__name__}")
        
        self.reranker = self.__class__._reranker_cache
    
    def execute(self, query: str, query_profile: QueryProfile, 
                doc_profile: Optional[DocumentProfile] = None) -> Dict[str, Any]:
        """Execute the strategy with multi-stage retrieval"""
        
        # Get optimal parameters
        params = self.get_params(query_profile, doc_profile)
        
        # Get strategy-specific config
        config_params = self._get_config_params(self.strategy_name) or {}
        
        # Multi-stage retrieval params
        retrieval_depth = config_params.get('retrieval_depth', 1)
        initial_candidates = config_params.get('initial_candidates', params.top_k)
        
        # Re-ranking params
        use_reranking = config_params.get('use_reranking', False)
        rerank_candidates = config_params.get('rerank_candidates', initial_candidates)
        
        logger.info(
            f"Executing {self.__class__.__name__} with top_k={params.top_k}, "
            f"depth={retrieval_depth}, reranking={use_reranking}"
        )
        
        # Stage 1: Retrieve using Hybrid Search (BM25 + Vector + RRF + Reranking)
        # We delegate all retrieval complexity to SimilaritySearch
        results = self.search.search(
            query, 
            top_k=params.top_k
        )
        
        if not results:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "context": "",
                "num_sources": 0,
                "confidence": 0.0
            }
        
        # Build context
        context = self._build_context(results)
        
        # Generate prompt
        prompt = self._generate_prompt(context, query, query_profile)
        
        # Generate answer
        answer = self.llm.generate(
            prompt, 
            temperature=params.temperature,
            max_tokens=params.max_tokens
        )
        
        # Extract sources
        sources = self._extract_sources(results)
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "num_sources": len(sources),
            "strategy": self.__class__.__name__,
            "params": params.__dict__,
            "used_reranking": use_reranking,
            "retrieval_depth": retrieval_depth
        }
    
    def _build_context(self, results: list) -> str:
        """Build context from search results"""
        texts = [r["metadata"].get("text", "") for r in results]
        return "\n\n".join(texts)
    
    def _extract_sources(self, results: list) -> list:
        """Extract source information"""
        sources = []
        for i, result in enumerate(results, 1):
            source_info = {
                "rank": i,
                "similarity_score": result.get("similarity_score", 0.0),
                "distance": result.get("distance", 0.0),
                "source": result["metadata"].get("source", "unknown"),
                "preview": result["metadata"].get("text", "")[:200] + "..."
            }
            sources.append(source_info)
        return sources
    
    @abstractmethod
    def _generate_prompt(self, context: str, query: str, 
                        query_profile: QueryProfile) -> str:
        """Generate prompt for this strategy"""
        pass
    
    @abstractmethod
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get optimal parameters for this query"""
        pass

"""
Base classes and interfaces for adaptive RAG system
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class QueryProfile:
    """Profile of a user query"""
    query: str
    query_type: str  # fact, summary, comparison, reasoning
    complexity: str  # simple, medium, complex
    entities: List[str]
    intent: str
    requires_multi_hop: bool
    estimated_chunks_needed: int


@dataclass
class DocumentProfile:
    """Profile of document collection"""
    doc_types: List[str]  # legal, technical, conversational
    has_tables: bool
    has_code: bool
    avg_section_length: int
    structure_type: str  # hierarchical, flat, mixed


@dataclass
class StrategyParams:
    """Parameters for a specific strategy"""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_depth: int  # How many stages
    use_reranking: bool
    use_query_expansion: bool
    temperature: float
    max_tokens: int


class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    @abstractmethod
    def execute(self, query: str, query_profile: QueryProfile, 
                doc_profile: Optional[DocumentProfile] = None) -> Dict[str, Any]:
        """Execute the strategy"""
        pass
    
    @abstractmethod
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get optimal parameters for this query"""
        pass


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    def get_cost_per_token(self) -> float:
        """Get cost per token (0 for local)"""
        return 0.0


class BaseEmbeddingProvider(ABC):
    """Abstract base class for embedding providers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Embed single text"""
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        """Embed batch of texts"""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        pass

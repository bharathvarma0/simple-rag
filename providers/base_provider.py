"""
Base provider classes
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np


class BaseLLMProvider(ABC):
    """Abstract base for LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate text from prompt"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    def get_cost_per_token(self) -> float:
        """Get cost per token (0 for local)"""
        return 0.0


class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding providers"""
    
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

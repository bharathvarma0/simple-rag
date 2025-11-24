"""
Ollama local LLM provider
"""

from providers.base_provider import BaseLLMProvider
import requests
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, model_name: str = "llama3.2", 
                 base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
        logger.info(f"Initialized Ollama provider with model: {model_name}")
    
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate using Ollama"""
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            available = response.status_code == 200
            if available:
                logger.info("Ollama is available")
            else:
                logger.warning("Ollama is not available")
            return available
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {e}")
            return False
    
    def get_cost_per_token(self) -> float:
        return 0.0  # Local is free

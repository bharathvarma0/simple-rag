"""
Groq API provider
"""

from providers.base_provider import BaseLLMProvider
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqProvider(BaseLLMProvider):
    """Groq API provider"""
    
    def __init__(self, model_name: str = "llama-3.1-70b-versatile", 
                 api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self._client = None
        logger.info(f"Initialized Groq provider with model: {model_name}")
    
    def _get_client(self):
        """Lazy load Groq client"""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
                logger.info("Groq client initialized")
            except ImportError:
                logger.error("Groq package not installed")
                raise ImportError("Groq package not installed. Run: pip install groq")
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate using Groq"""
        
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise RuntimeError(f"Groq generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Groq API key is valid"""
        if not self.api_key:
            logger.warning("Groq API key not provided")
            return False
        try:
            client = self._get_client()
            # Simple test call
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info("Groq is available")
            return True
        except Exception as e:
            logger.warning(f"Groq availability check failed: {e}")
            return False
    
    def get_cost_per_token(self) -> float:
        # Groq pricing (approximate)
        return 0.00001  # $0.01 per 1K tokens

"""
OpenAI API provider
"""

from providers.base_provider import BaseLLMProvider
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, model_name: str = "gpt-4-turbo-preview", 
                 api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self._client = None
        logger.info(f"Initialized OpenAI provider with model: {model_name}")
    
    def _get_client(self):
        """Lazy load OpenAI client"""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError:
                logger.error("OpenAI package not installed")
                raise ImportError("OpenAI package not installed. Run: pip install openai")
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate using OpenAI"""
        
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
            logger.error(f"OpenAI generation failed: {e}")
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI API key is valid"""
        if not self.api_key:
            logger.warning("OpenAI API key not provided")
            return False
        try:
            client = self._get_client()
            # Simple test call
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info("OpenAI is available")
            return True
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            return False
    
    def get_cost_per_token(self) -> float:
        # OpenAI pricing (approximate for GPT-4)
        return 0.00003  # $0.03 per 1K tokens

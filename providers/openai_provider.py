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
    
    def __init__(self, model_name: str = "gpt-5-nano", 
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
        
        # Handle parameter differences for newer models (o1, gpt-5)
        # These models often don't support 'max_tokens' or 'temperature' in the same way
        kwargs = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        is_new_model = any(x in self.model_name for x in ["gpt-5", "o1-", "o3-"])
        
        if is_new_model:
            # Newer models use max_completion_tokens
            kwargs["max_completion_tokens"] = max_tokens
            # gpt-5-nano and o1 models do NOT support temperature
            # kwargs["temperature"] = temperature 
        else:
            # Standard models
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature
        
        try:
            response = client.chat.completions.create(**kwargs)
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
            
            # Prepare test call args
            kwargs = {
                "model": self.model_name,
                "messages": [{"role": "user", "content": "test"}]
            }
            
            if any(x in self.model_name for x in ["gpt-5", "o1-", "o3-"]):
                kwargs["max_completion_tokens"] = 1
            else:
                kwargs["max_tokens"] = 1
                
            # Simple test call
            client.chat.completions.create(**kwargs)
            logger.info("OpenAI is available")
            return True
        except Exception as e:
            logger.warning(f"OpenAI availability check failed: {e}")
            return False
    
    def get_cost_per_token(self) -> float:
        # OpenAI pricing (approximate for GPT-4)
        return 0.00003  # $0.03 per 1K tokens

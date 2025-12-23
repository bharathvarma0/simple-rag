"""
LLM wrapper for different providers
"""

import os
from typing import Optional, List, Dict, Any
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from config import get_settings


class LLMWrapper:
    """Wrapper for LLM providers"""
    
    def __init__(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize LLM wrapper
        
        Args:
            provider: LLM provider ("groq" or "openai")
            model_name: Model name (uses config default if None)
            api_key: API key (uses env var if None)
            temperature: Temperature (uses config default if None)
            max_tokens: Max tokens (uses config default if None)
        """
        self.settings = get_settings()
        
        self.provider = provider or self.settings.llm.provider
        self.model_name = model_name or self.settings.llm.model_name
        self.api_key = api_key or self.settings.llm.api_key
        self.temperature = temperature if temperature is not None else self.settings.llm.temperature
        self.max_tokens = max_tokens or self.settings.llm.max_tokens
        
        self._llm = None
        self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize LLM client"""
        if not self.api_key:
            raise ValueError(
                f"{self.provider.upper()}_API_KEY not found. "
                "Set it as environment variable or pass as parameter."
            )
        
        if self.provider.lower() == "groq":
            self._llm = ChatGroq(
                groq_api_key=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        elif self.provider.lower() == "openai":
            self._llm = ChatOpenAI(
                openai_api_key=self.api_key,
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
        print(f"[INFO] Initialized {self.provider} LLM with model: {self.model_name}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Generate text from prompt
        
        Args:
            prompt: Input prompt
            **kwargs: Additional arguments for LLM
            
        Returns:
            Generated text
        """
        if self._llm is None:
            raise ValueError("LLM not initialized")
        
        # Handle both string and list prompts
        if isinstance(prompt, str):
            messages = [prompt]
        else:
            messages = prompt
        
        response = self._llm.invoke(messages, **kwargs)
        return response.content
    
    def invoke(self, messages: List[str], **kwargs) -> Any:
        """
        Invoke LLM with messages
        
        Args:
            messages: List of message strings
            **kwargs: Additional arguments
            
        Returns:
            LLM response
        """
        if self._llm is None:
            raise ValueError("LLM not initialized")
        
        return self._llm.invoke(messages, **kwargs)
    
    @property
    def llm(self):
        """Get underlying LLM client"""
        return self._llm



"""
Adaptive RAG Pipeline - Main entry point for version 2
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional
import time

sys.path.append(str(Path(__file__).parent))

from core.query_analyzer import QueryAnalyzer
from core.strategy_selector import StrategySelector
from core.metrics import MetricsTracker
from strategies import (
    SimpleFactStrategy,
    ComplexFactStrategy,
    SummaryStrategy,
    ComparisonStrategy,
    ReasoningStrategy
)
from providers.ollama_provider import OllamaProvider
from providers.groq_provider import GroqProvider
from providers.openai_provider import OpenAIProvider
from augmentation.vector_db import VectorDatabase
from utils.logger import get_logger
from config import get_settings
import os
import yaml

from utils.cache import cache_result

logger = get_logger(__name__)


class AdaptiveRAGPipeline:
    """Adaptive RAG pipeline with strategy-based routing"""
    
    def __init__(self, persist_dir: Optional[str] = None):
        """Initialize adaptive RAG pipeline"""
        
        self.settings = get_settings()
        self.persist_dir = persist_dir or self.settings.vector_store.persist_dir
        
        logger.info("=" * 60)
        logger.info("Initializing Adaptive RAG Pipeline v2")
        logger.info("=" * 60)
        
        # Load provider configuration
        self.provider_config = self._load_provider_config()
        
        # Initialize LLM provider
        self.llm = self._initialize_llm()
        
        # Initialize vector database
        self.vector_db = VectorDatabase(
            persist_dir=self.persist_dir,
            embedding_model=self.settings.embedding.model_name
        )
        
        # Load vector database
        if self.vector_db.exists():
            self.vector_db.load()
        else:
            # Don't raise error here, might be starting fresh
            logger.warning(
                f"Vector database not found at {self.persist_dir}. "
                "Please run data ingestion."
            )
        
        # Initialize query analyzer
        self.query_analyzer = QueryAnalyzer(self.llm)
        
        # Initialize strategy selector
        self.strategy_selector = StrategySelector()
        
        # Register strategies
        self._register_strategies()
        
        # Initialize metrics tracker
        self.metrics = MetricsTracker()
        
        logger.info("Adaptive RAG Pipeline initialized successfully")
        logger.info(f"LLM Provider: {self.llm.__class__.__name__}")
        logger.info(f"Available strategies: {self.strategy_selector.get_available_strategies()}")
    
    def _load_provider_config(self) -> Dict:
        """Load provider configuration from YAML"""
        config_path = Path(__file__).parent / "config" / "providers.yaml"
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            logger.warning("Provider config not found, using defaults")
            return {'providers': {'llm': {'priority': ['ollama']}}}
    
    def _initialize_llm(self):
        """Initialize LLM provider based on configuration and availability"""
        
        llm_config = self.provider_config.get('providers', {}).get('llm', {})
        priority = llm_config.get('priority', ['ollama'])
        
        for provider_name in priority:
            provider_settings = llm_config.get(provider_name, {})
            
            if not provider_settings.get('enabled', True):
                logger.info(f"Provider {provider_name} is disabled, skipping")
                continue
            
            try:
                if provider_name == 'ollama':
                    provider = OllamaProvider(
                        model_name=provider_settings.get('model', 'llama3.2'),
                        base_url=provider_settings.get('base_url', 'http://localhost:11434')
                    )
                elif provider_name == 'groq':
                    api_key = os.getenv(provider_settings.get('api_key_env', 'GROQ_API_KEY'))
                    provider = GroqProvider(
                        model_name=provider_settings.get('model', 'llama-3.1-70b-versatile'),
                        api_key=api_key
                    )
                elif provider_name == 'openai':
                    api_key = os.getenv(provider_settings.get('api_key_env', 'OPENAI_API_KEY'))
                    provider = OpenAIProvider(
                        model_name=provider_settings.get('model', 'gpt-4-turbo-preview'),
                        api_key=api_key
                    )
                else:
                    logger.warning(f"Unknown provider: {provider_name}")
                    continue
                
                # Check if provider is available
                if provider.is_available():
                    logger.info(f"Successfully initialized {provider_name} provider")
                    return provider
                else:
                    logger.warning(f"Provider {provider_name} not available, trying next")
            
            except Exception as e:
                logger.warning(f"Failed to initialize {provider_name}: {e}")
                continue
        
        raise RuntimeError("No LLM provider available. Please configure at least one provider.")
    
    def _register_strategies(self):
        """Register all available strategies"""
        
        strategies = {
            'simple_fact': SimpleFactStrategy,
            'complex_fact': ComplexFactStrategy,
            'summary': SummaryStrategy,
            'comparison': ComparisonStrategy,
            'reasoning': ReasoningStrategy
        }
        
        for name, strategy_class in strategies.items():
            self.strategy_selector.register_strategy(name, strategy_class)
    
    @cache_result(prefix="rag_query", ttl=3600)
    def query(self, question: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the adaptive RAG system
        
        Args:
            question: User question
            doc_id: Optional document ID to filter by
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        
        start_time = time.time()
        
        logger.info("=" * 60)
        logger.info(f"Processing query: {question} (doc_id={doc_id})")
        logger.info("=" * 60)
        
        # Step 1: Analyze query
        query_profile = self.query_analyzer.analyze(question)
        
        # Step 2: Select strategy
        strategy = self.strategy_selector.select(query_profile)
        
        # Step 3: Initialize strategy with vector DB and LLM
        strategy.initialize(self.vector_db, self.llm)
        
        # Step 4: Execute strategy
        result = strategy.execute(question, query_profile, doc_id=doc_id)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log metrics
        self.metrics.log_query(
            query=question,
            query_profile=query_profile.__dict__,
            strategy_used=result.get('strategy', 'unknown'),
            response_time=response_time,
            result=result
        )
        
        # Add metadata to result
        result['query_profile'] = query_profile.__dict__
        result['response_time'] = response_time
        
        logger.info(f"Query completed in {response_time:.2f}s")
        logger.info(f"Strategy used: {result.get('strategy')}")
        logger.info(f"Sources retrieved: {result.get('num_sources')}")
        
        return result
    
    def ask(self, question: str) -> str:
        """
        Simple query method that returns only the answer
        
        Args:
            question: User question
            
        Returns:
            Answer string
        """
        result = self.query(question)
        return result["answer"]
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for current session"""
        return self.metrics.get_session_stats()

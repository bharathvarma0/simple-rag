"""
Strategy selection and routing module
"""

from typing import Dict, Type, Optional
from core.base import BaseStrategy, QueryProfile, DocumentProfile
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class StrategySelector:
    """Select optimal strategy based on query and document profiles"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.strategy_instances: Dict[str, BaseStrategy] = {}
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy"""
        self.strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    def select(self, query_profile: QueryProfile, 
               doc_profile: Optional[DocumentProfile] = None) -> BaseStrategy:
        """Select best strategy for this query"""
        
        # Determine strategy name
        strategy_name = self._determine_strategy(query_profile, doc_profile)
        
        logger.info(f"Selected strategy: {strategy_name} for query type: {query_profile.query_type}")
        
        # Get or create strategy instance
        if strategy_name not in self.strategy_instances:
            strategy_class = self.strategies.get(strategy_name)
            if not strategy_class:
                # Fallback to first available strategy
                logger.warning(f"Strategy '{strategy_name}' not found, using fallback")
                strategy_name = list(self.strategies.keys())[0]
                strategy_class = self.strategies[strategy_name]
            
            self.strategy_instances[strategy_name] = strategy_class()
        
        return self.strategy_instances[strategy_name]
    
    def _determine_strategy(self, query_profile: QueryProfile, 
                           doc_profile: Optional[DocumentProfile]) -> str:
        """Determine which strategy to use"""
        
        query_type = query_profile.query_type
        complexity = query_profile.complexity
        
        # Strategy selection logic
        if query_type == 'fact':
            if complexity == 'simple':
                return 'simple_fact'
            else:
                return 'complex_fact'
        
        elif query_type == 'summary':
            return 'summary'
        
        elif query_type == 'comparison':
            return 'comparison'
        
        elif query_type == 'reasoning':
            return 'reasoning'
        
        else:
            # Default to simple fact
            return 'simple_fact'
    
    def get_available_strategies(self) -> list:
        """Get list of available strategies"""
        return list(self.strategies.keys())

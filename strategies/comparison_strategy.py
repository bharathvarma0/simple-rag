"""
Comparison strategy
"""

from strategies.base_strategy import BaseStrategy
from core.base import QueryProfile, DocumentProfile, StrategyParams
from typing import Optional


class ComparisonStrategy(BaseStrategy):
    """Strategy for comparison questions"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = 'comparison'
    
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get parameters for comparison"""
        
        # Try to load from YAML config first
        config = self._get_config_params('comparison')
        
        if config:
            # Use YAML configuration
            return StrategyParams(
                chunk_size=config.get('chunk_size', 1500),
                chunk_overlap=config.get('chunk_overlap', 300),
                top_k=config.get('top_k', 20),
                retrieval_depth=config.get('retrieval_depth', 2),
                use_reranking=config.get('use_reranking', True),
                use_query_expansion=config.get('use_query_expansion', True),
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 1500)
            )
        
        # Fallback to hardcoded defaults
        return StrategyParams(
            chunk_size=1500,
            chunk_overlap=300,
            top_k=20,
            retrieval_depth=2,
            use_reranking=True,
            use_query_expansion=True,
            temperature=0.1,
            max_tokens=1500
        )
    
    def _generate_prompt(self, context: str, query: str, 
                        query_profile: QueryProfile) -> str:
        """Generate prompt for comparison"""
        return f"""You are comparing information from regulatory documents. Make accurate, structured comparisons.

Context:
{context}

Question: {query}

Instructions:
1. Identify the items being compared
2. Extract relevant information for EACH item from the context
3. Present the comparison in a clear, structured format (e.g., table or bullet points)
4. Highlight similarities and differences
5. Cite specific sections for each point
6. If information is missing for any item, explicitly state what's missing
7. Do NOT make assumptions or use external knowledge

Comparison:"""

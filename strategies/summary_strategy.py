"""
Summarization strategy
"""

from strategies.base_strategy import BaseStrategy
from core.base import QueryProfile, DocumentProfile, StrategyParams
from typing import Optional


class SummaryStrategy(BaseStrategy):
    """Strategy for summarization questions"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = 'summary'
    
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get parameters for summarization"""
        
        # Try to load from YAML config first
        config = self._get_config_params('summary')
        
        if config:
            # Use YAML configuration
            return StrategyParams(
                chunk_size=config.get('chunk_size', 2000),
                chunk_overlap=config.get('chunk_overlap', 400),
                top_k=config.get('top_k', 30),
                retrieval_depth=config.get('retrieval_depth', 1),
                use_reranking=config.get('use_reranking', True),
                use_query_expansion=config.get('use_query_expansion', False),
                temperature=config.get('temperature', 0.2),
                max_tokens=config.get('max_tokens', 2048)
            )
        
        # Fallback to hardcoded defaults
        return StrategyParams(
            chunk_size=2000,
            chunk_overlap=400,
            top_k=30,
            retrieval_depth=1,
            use_reranking=True,
            use_query_expansion=False,
            temperature=0.2,
            max_tokens=2048
        )
    
    def _generate_prompt(self, context: str, query: str, 
                        query_profile: QueryProfile) -> str:
        """Generate prompt for summarization"""
        return f"""You are summarizing information from regulatory documents. Create a comprehensive yet concise summary.

Context:
{context}

Question: {query}

Instructions:
1. Read all provided context sections
2. Identify the main topics and key points
3. Organize the information logically
4. Create a structured summary with clear sections
5. Include specific details (numbers, dates, requirements) where relevant
6. Cite sources for major points
7. Do NOT add information not present in the context

Summary:"""

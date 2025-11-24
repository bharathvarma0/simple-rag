"""
Reasoning strategy
"""

from strategies.base_strategy import BaseStrategy
from core.base import QueryProfile, DocumentProfile, StrategyParams
from typing import Optional


class ReasoningStrategy(BaseStrategy):
    """Strategy for complex reasoning questions"""
    
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get parameters for reasoning"""
        
        # Try to load from YAML config first
        config = self._get_config_params('reasoning')
        
        if config:
            # Use YAML configuration
            return StrategyParams(
                chunk_size=config.get('chunk_size', 1800),
                chunk_overlap=config.get('chunk_overlap', 350),
                top_k=config.get('top_k', 15),
                retrieval_depth=config.get('retrieval_depth', 2),
                use_reranking=config.get('use_reranking', True),
                use_query_expansion=config.get('use_query_expansion', True),
                temperature=config.get('temperature', 0.15),
                max_tokens=config.get('max_tokens', 2048)
            )
        
        # Fallback to hardcoded defaults
        return StrategyParams(
            chunk_size=1800,
            chunk_overlap=350,
            top_k=15,
            retrieval_depth=2,
            use_reranking=True,
            use_query_expansion=True,
            temperature=0.15,
            max_tokens=2048
        )
    
    def _generate_prompt(self, context: str, query: str, 
                        query_profile: QueryProfile) -> str:
        """Generate prompt for reasoning"""
        return f"""You are analyzing complex regulatory information. Think step-by-step and reason carefully.

Context:
{context}

Question: {query}

Instructions:
1. Break down the question into sub-questions if needed
2. For each sub-question, find relevant information in the context
3. Reason through the connections between different pieces of information
4. Think step-by-step and show your reasoning process
5. Synthesize a comprehensive answer
6. Cite all sources used in your reasoning
7. If you need to make inferences, clearly state them as such
8. Do NOT use external knowledge - only reason from the provided context

Step-by-step analysis:"""

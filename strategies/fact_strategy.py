"""
Fact retrieval strategies
"""

from strategies.base_strategy import BaseStrategy
from core.base import QueryProfile, DocumentProfile, StrategyParams
from typing import Optional


class SimpleFactStrategy(BaseStrategy):
    """Strategy for simple fact retrieval questions"""
    
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get parameters for simple fact retrieval"""
        
        # Try to load from YAML config first
        config = self._get_config_params('simple_fact')
        
        if config:
            # Use YAML configuration
            return StrategyParams(
                chunk_size=config.get('chunk_size', 800),
                chunk_overlap=config.get('chunk_overlap', 150),
                top_k=config.get('top_k', 3),
                retrieval_depth=config.get('retrieval_depth', 1),
                use_reranking=config.get('use_reranking', False),
                use_query_expansion=config.get('use_query_expansion', False),
                temperature=config.get('temperature', 0.0),
                max_tokens=config.get('max_tokens', 500)
            )
        
        # Fallback to hardcoded defaults
        return StrategyParams(
            chunk_size=800,
            chunk_overlap=150,
            top_k=3,
            retrieval_depth=1,
            use_reranking=False,
            use_query_expansion=False,
            temperature=0.0,
            max_tokens=500
        )
    
    def _generate_prompt(self, context: str, query: str, 
                        query_profile: QueryProfile) -> str:
        """Generate prompt for simple fact retrieval"""
        return f"""You are a precise information retrieval assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {query}

Instructions:
1. Read the context carefully
2. Identify the specific information that answers the question
3. If the answer is in the context, provide it with the exact reference (e.g., Article number, section)
4. If the answer is NOT in the context, say "I cannot find this information in the provided context"
5. Do NOT use external knowledge or make assumptions
6. Be concise and direct

Answer:"""


class ComplexFactStrategy(BaseStrategy):
    """Strategy for complex fact retrieval requiring multiple sources"""
    
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: Optional[DocumentProfile] = None) -> StrategyParams:
        """Get parameters for complex fact retrieval"""
        
        # Try to load from YAML config first
        config = self._get_config_params('complex_fact')
        
        if config:
            # Use YAML configuration
            return StrategyParams(
                chunk_size=config.get('chunk_size', 1200),
                chunk_overlap=config.get('chunk_overlap', 250),
                top_k=config.get('top_k', 10),
                retrieval_depth=config.get('retrieval_depth', 2),
                use_reranking=config.get('use_reranking', True),
                use_query_expansion=config.get('use_query_expansion', True),
                temperature=config.get('temperature', 0.1),
                max_tokens=config.get('max_tokens', 1000)
            )
        
        # Fallback to hardcoded defaults
        return StrategyParams(
            chunk_size=1200,
            chunk_overlap=250,
            top_k=10,
            retrieval_depth=2,
            use_reranking=True,
            use_query_expansion=True,
            temperature=0.1,
            max_tokens=1000
        )
    
    def _generate_prompt(self, context: str, query: str, 
                        query_profile: QueryProfile) -> str:
        """Generate prompt for complex fact retrieval"""
        return f"""You are analyzing regulatory documents to answer a detailed question. Use the provided context carefully.

Context:
{context}

Question: {query}

Instructions:
1. Read all context sections carefully
2. Identify ALL relevant information that relates to the question
3. If the information is spread across multiple sections, synthesize it
4. Cite specific sections or articles for each piece of information
5. If any part of the answer is missing, explicitly state what's missing
6. Do NOT use external knowledge

Detailed answer with citations:"""

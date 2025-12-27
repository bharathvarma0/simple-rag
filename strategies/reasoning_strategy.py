"""
Reasoning strategy with chain-of-thought prompting
"""

from strategies.base_strategy import BaseStrategy
from core.base import QueryProfile, DocumentProfile, StrategyParams
from typing import Optional


class ReasoningStrategy(BaseStrategy):
    """Strategy for complex reasoning questions"""
    
    def __init__(self):
        super().__init__()
        self.strategy_name = 'reasoning'
    
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
        """Generate chain-of-thought reasoning prompt"""
        
        prompt = f"""You are tasked with answering a complex question that requires multi-step reasoning and logical inference.

Context Information:
{context}

Question: {query}

Please analyze this question step-by-step using the following structure:

Step 1 - Identify Key Facts:
First, identify all relevant facts from the context that relate to the question.

Step 2 - Analyze Relationships:
Next, analyze how these facts connect to each other and to the question.

Step 3 - Apply Logic:
Apply logical reasoning to draw conclusions from the facts and their relationships.

Step 4 - Synthesize Answer:
Finally, synthesize your reasoning into a clear, comprehensive answer.

Important: Base your answer ONLY on the provided context. Do not use external knowledge.

Your detailed answer with reasoning:
"""
        
        return prompt

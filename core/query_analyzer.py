"""
Query analysis and classification module
"""

from typing import Dict, Any, List
from core.base import QueryProfile, BaseLLMProvider
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class QueryAnalyzer:
    """Analyze and classify user queries"""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider
        self.classification_cache = {}
    
    def analyze(self, query: str) -> QueryProfile:
        """Analyze query and return profile"""
        
        # Check cache first
        if query in self.classification_cache:
            logger.info(f"Using cached classification for query")
            return self.classification_cache[query]
        
        logger.info(f"Analyzing query: {query[:50]}...")
        
        # Use LLM to classify query
        classification = self._classify_with_llm(query)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Determine complexity
        complexity = self._assess_complexity(query, classification)
        
        # Estimate chunks needed
        chunks_needed = self._estimate_chunks(classification, complexity)
        
        profile = QueryProfile(
            query=query,
            query_type=classification['type'],
            complexity=complexity,
            entities=entities,
            intent=classification['intent'],
            requires_multi_hop=classification['multi_hop'],
            estimated_chunks_needed=chunks_needed
        )
        
        logger.info(f"Query classified as: {profile.query_type} ({profile.complexity}), "
                   f"estimated chunks: {profile.estimated_chunks_needed}")
        
        # Cache result
        self.classification_cache[query] = profile
        
        return profile
    
    def _classify_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to classify query type and intent"""
        
        prompt = f"""Analyze this query and classify it:

Query: {query}

Provide classification in this exact format:
TYPE: [fact|summary|comparison|reasoning]
INTENT: [search|explain|compare|analyze]
MULTI_HOP: [yes|no]

Guidelines:
- fact: Simple factual question (What is X? Who is Y?)
- summary: Asking for overview or summary (Summarize X, What are the main points?)
- comparison: Comparing multiple items (Compare X and Y, What's the difference?)
- reasoning: Requires analysis or inference (How does X affect Y? Why would X happen?)

Classification:"""
        
        try:
            response = self.llm.generate(prompt, temperature=0.0, max_tokens=150)
            
            # Parse response
            classification = {
                'type': self._extract_field(response, 'TYPE', 'fact'),
                'intent': self._extract_field(response, 'INTENT', 'search'),
                'multi_hop': self._extract_field(response, 'MULTI_HOP', 'no') == 'yes'
            }
            
            return classification
        except Exception as e:
            logger.warning(f"LLM classification failed: {e}, using heuristics")
            return self._classify_with_heuristics(query)
    
    def _classify_with_heuristics(self, query: str) -> Dict[str, Any]:
        """Fallback: classify using simple heuristics"""
        
        query_lower = query.lower()
        
        # Detect query type
        if any(word in query_lower for word in ['summarize', 'overview', 'main points', 'key changes']):
            query_type = 'summary'
        elif any(word in query_lower for word in ['compare', 'difference', 'versus', 'vs']):
            query_type = 'comparison'
        elif any(word in query_lower for word in ['how does', 'why', 'explain', 'what is the impact']):
            query_type = 'reasoning'
        else:
            query_type = 'fact'
        
        # Detect intent
        if 'compare' in query_lower:
            intent = 'compare'
        elif any(word in query_lower for word in ['explain', 'how', 'why']):
            intent = 'explain'
        elif 'analyze' in query_lower:
            intent = 'analyze'
        else:
            intent = 'search'
        
        # Detect multi-hop
        multi_hop = any(word in query_lower for word in ['affect', 'impact', 'result in', 'lead to'])
        
        return {
            'type': query_type,
            'intent': intent,
            'multi_hop': multi_hop
        }
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query"""
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]*)"', query)
        quoted.extend(re.findall(r"'([^']*)'", query))
        
        # Extract capitalized terms (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        
        # Extract technical terms (hyphenated, acronyms)
        technical = re.findall(r'\b[A-Z]{2,}\b', query)  # Acronyms
        technical.extend(re.findall(r'\b\w+-\w+\b', query))  # Hyphenated
        
        entities = list(set(quoted + capitalized + numbers + technical))
        return entities
    
    def _assess_complexity(self, query: str, classification: Dict) -> str:
        """Assess query complexity"""
        
        # Simple heuristics
        word_count = len(query.split())
        has_multiple_questions = query.count('?') > 1
        
        if classification['multi_hop'] or has_multiple_questions:
            return 'complex'
        elif word_count > 15 or classification['type'] in ['comparison', 'reasoning']:
            return 'medium'
        else:
            return 'simple'
    
    def _estimate_chunks(self, classification: Dict, complexity: str) -> int:
        """Estimate how many chunks are needed"""
        
        base_chunks = {
            'fact': 5,
            'summary': 20,
            'comparison': 15,
            'reasoning': 12
        }
        
        multiplier = {
            'simple': 0.6,
            'medium': 1.0,
            'complex': 1.5
        }
        
        query_type = classification['type']
        base = base_chunks.get(query_type, 10)
        mult = multiplier.get(complexity, 1.0)
        
        return int(base * mult)
    
    def _extract_field(self, text: str, field: str, default: str) -> str:
        """Extract field from LLM response"""
        pattern = f"{field}:\\s*\\[?([^\\]\\n]+)\\]?"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        return default

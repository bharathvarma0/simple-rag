# Adaptive RAG System - Implementation Plan

## Goal
Build a production-ready, adaptive RAG system that:
1. Works efficiently with local models (Ollama + local embeddings)
2. Easily migrates to paid APIs (Groq, OpenAI) without code rewrites
3. Adapts parameters based on query type, document type, and user feedback
4. Achieves 75-80% accuracy on your evaluation set

---

## Architecture Principles

### 1. **Abstraction Layers** (API-Agnostic)
```
User Query
    ↓
Query Understanding Layer (API-agnostic)
    ↓
Strategy Selection Layer (API-agnostic)
    ↓
Retrieval Layer (Model-agnostic)
    ↓
Generation Layer (LLM Provider abstraction)
    ↓
Validation Layer (API-agnostic)
    ↓
Response
```

### 2. **Modular Components**
- Each component is independent and swappable
- Configuration-driven (not hardcoded)
- Easy to A/B test different implementations

### 3. **Future-Proof Design**
- Provider abstraction for LLMs and embeddings
- Strategy pattern for different query types
- Plugin architecture for new features

---

## Detailed Implementation Plan

## Phase 1: Foundation & Architecture (Week 1)

### 1.1 Create Modular Architecture

**Files to Create:**
```
rag-project/
├── core/
│   ├── __init__.py
│   ├── base.py              # Base classes and interfaces
│   ├── query_analyzer.py    # Query understanding
│   ├── strategy_selector.py # Route queries to strategies
│   └── metrics.py           # Performance tracking
├── strategies/
│   ├── __init__.py
│   ├── base_strategy.py     # Abstract strategy class
│   ├── fact_strategy.py     # For simple fact retrieval
│   ├── summary_strategy.py  # For summarization
│   ├── comparison_strategy.py # For comparisons
│   └── reasoning_strategy.py  # For complex reasoning
├── providers/
│   ├── __init__.py
│   ├── base_provider.py     # Abstract provider interface
│   ├── ollama_provider.py   # Local Ollama
│   ├── groq_provider.py     # Groq API
│   └── openai_provider.py   # OpenAI API
└── config/
    ├── strategies.yaml      # Strategy configurations
    └── providers.yaml       # Provider configurations
```

**Key Implementation:**

#### `core/base.py` - Base Interfaces
```python
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class QueryProfile:
    """Profile of a user query"""
    query: str
    query_type: str  # fact, summary, comparison, reasoning
    complexity: str  # simple, medium, complex
    entities: List[str]
    intent: str
    requires_multi_hop: bool
    estimated_chunks_needed: int

@dataclass
class DocumentProfile:
    """Profile of document collection"""
    doc_types: List[str]  # legal, technical, conversational
    has_tables: bool
    has_code: bool
    avg_section_length: int
    structure_type: str  # hierarchical, flat, mixed

@dataclass
class StrategyParams:
    """Parameters for a specific strategy"""
    chunk_size: int
    chunk_overlap: int
    top_k: int
    retrieval_depth: int  # How many stages
    use_reranking: bool
    use_query_expansion: bool
    temperature: float
    max_tokens: int

class BaseStrategy(ABC):
    """Abstract base class for all strategies"""
    
    @abstractmethod
    def execute(self, query: str, query_profile: QueryProfile, 
                doc_profile: DocumentProfile) -> Dict[str, Any]:
        """Execute the strategy"""
        pass
    
    @abstractmethod
    def get_params(self, query_profile: QueryProfile, 
                   doc_profile: DocumentProfile) -> StrategyParams:
        """Get optimal parameters for this query"""
        pass

class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float, 
                 max_tokens: int) -> str:
        """Generate response"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    @abstractmethod
    def get_cost_per_token(self) -> float:
        """Get cost per token (0 for local)"""
        pass
```

#### `core/query_analyzer.py` - Query Understanding
```python
from typing import Dict, Any
from core.base import QueryProfile, BaseLLMProvider
import re

class QueryAnalyzer:
    """Analyze and classify user queries"""
    
    def __init__(self, llm_provider: BaseLLMProvider):
        self.llm = llm_provider
        self.classification_cache = {}
    
    def analyze(self, query: str) -> QueryProfile:
        """Analyze query and return profile"""
        
        # Check cache first
        if query in self.classification_cache:
            return self.classification_cache[query]
        
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
EXPLANATION: [brief explanation]

Classification:"""
        
        response = self.llm.generate(prompt, temperature=0.0, max_tokens=150)
        
        # Parse response
        classification = {
            'type': self._extract_field(response, 'TYPE', 'fact'),
            'intent': self._extract_field(response, 'INTENT', 'search'),
            'multi_hop': self._extract_field(response, 'MULTI_HOP', 'no') == 'yes'
        }
        
        return classification
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from query"""
        # Simple regex-based extraction (can be improved with NER)
        
        # Extract quoted terms
        quoted = re.findall(r'"([^"]*)"', query)
        quoted.extend(re.findall(r"'([^']*)'", query))
        
        # Extract capitalized terms (potential entities)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        
        entities = list(set(quoted + capitalized + numbers))
        return entities
    
    def _assess_complexity(self, query: str, classification: Dict) -> str:
        """Assess query complexity"""
        
        # Simple heuristics
        word_count = len(query.split())
        has_multiple_questions = '?' in query[:-1]  # Multiple question marks
        
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
```

#### `core/strategy_selector.py` - Strategy Router
```python
from typing import Dict, Type
from core.base import BaseStrategy, QueryProfile, DocumentProfile

class StrategySelector:
    """Select optimal strategy based on query and document profiles"""
    
    def __init__(self):
        self.strategies: Dict[str, Type[BaseStrategy]] = {}
        self.strategy_cache = {}
    
    def register_strategy(self, name: str, strategy_class: Type[BaseStrategy]):
        """Register a new strategy"""
        self.strategies[name] = strategy_class
    
    def select(self, query_profile: QueryProfile, 
               doc_profile: DocumentProfile) -> BaseStrategy:
        """Select best strategy for this query"""
        
        # Create cache key
        cache_key = f"{query_profile.query_type}_{query_profile.complexity}"
        
        # Check cache
        if cache_key in self.strategy_cache:
            return self.strategy_cache[cache_key]
        
        # Select strategy based on query type
        strategy_name = self._determine_strategy(query_profile, doc_profile)
        
        # Instantiate strategy
        strategy_class = self.strategies.get(strategy_name)
        if not strategy_class:
            # Fallback to default
            strategy_class = self.strategies.get('fact', list(self.strategies.values())[0])
        
        strategy = strategy_class()
        
        # Cache for reuse
        self.strategy_cache[cache_key] = strategy
        
        return strategy
    
    def _determine_strategy(self, query_profile: QueryProfile, 
                           doc_profile: DocumentProfile) -> str:
        """Determine which strategy to use"""
        
        query_type = query_profile.query_type
        complexity = query_profile.complexity
        
        # Simple mapping (can be made more sophisticated)
        if query_type == 'fact' and complexity == 'simple':
            return 'simple_fact'
        elif query_type == 'fact' and complexity in ['medium', 'complex']:
            return 'complex_fact'
        elif query_type == 'summary':
            return 'summary'
        elif query_type == 'comparison':
            return 'comparison'
        elif query_type == 'reasoning':
            return 'reasoning'
        else:
            return 'fact'  # Default
```

### 1.2 Implement Provider Abstraction

#### `providers/base_provider.py`
```python
from abc import ABC, abstractmethod
from typing import Optional, List
import numpy as np

class BaseLLMProvider(ABC):
    """Abstract base for LLM providers"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
    
    @abstractmethod
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass
    
    def get_cost_per_token(self) -> float:
        return 0.0  # Override in paid providers

class BaseEmbeddingProvider(ABC):
    """Abstract base for embedding providers"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
    
    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        pass
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> np.ndarray:
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        pass
```

#### `providers/ollama_provider.py`
```python
from providers.base_provider import BaseLLMProvider
import requests
import json

class OllamaProvider(BaseLLMProvider):
    """Ollama local LLM provider"""
    
    def __init__(self, model_name: str = "llama3.2", 
                 base_url: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.base_url = base_url
    
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate using Ollama"""
        
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": temperature,
            "num_predict": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def get_cost_per_token(self) -> float:
        return 0.0  # Local is free
```

#### `providers/groq_provider.py`
```python
from providers.base_provider import BaseLLMProvider
from typing import Optional

class GroqProvider(BaseLLMProvider):
    """Groq API provider"""
    
    def __init__(self, model_name: str = "llama-3.1-70b-versatile", 
                 api_key: Optional[str] = None):
        super().__init__(model_name, api_key)
        self._client = None
    
    def _get_client(self):
        """Lazy load Groq client"""
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Groq package not installed. Run: pip install groq")
        return self._client
    
    def generate(self, prompt: str, temperature: float = 0.1, 
                 max_tokens: int = 2048) -> str:
        """Generate using Groq"""
        
        client = self._get_client()
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"Groq generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Groq API key is valid"""
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            # Simple test call
            client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except:
            return False
    
    def get_cost_per_token(self) -> float:
        # Groq pricing (approximate)
        return 0.00001  # $0.01 per 1K tokens
```

### 1.3 Configuration System

#### `config/strategies.yaml`
```yaml
strategies:
  simple_fact:
    chunk_size: 800
    chunk_overlap: 150
    top_k: 3
    retrieval_depth: 1
    use_reranking: false
    use_query_expansion: false
    temperature: 0.0
    max_tokens: 500
    
  complex_fact:
    chunk_size: 1200
    chunk_overlap: 250
    top_k: 10
    retrieval_depth: 2
    use_reranking: true
    use_query_expansion: true
    temperature: 0.1
    max_tokens: 1000
    
  summary:
    chunk_size: 2000
    chunk_overlap: 400
    top_k: 30
    retrieval_depth: 1
    use_reranking: false
    use_query_expansion: false
    temperature: 0.2
    max_tokens: 2048
    
  comparison:
    chunk_size: 1500
    chunk_overlap: 300
    top_k: 20
    retrieval_depth: 2
    use_reranking: true
    use_query_expansion: true
    temperature: 0.1
    max_tokens: 1500
    
  reasoning:
    chunk_size: 1800
    chunk_overlap: 350
    top_k: 15
    retrieval_depth: 2
    use_reranking: true
    use_query_expansion: true
    temperature: 0.15
    max_tokens: 2048
```

#### `config/providers.yaml`
```yaml
providers:
  llm:
    # Priority order - system will try in order until one works
    priority:
      - ollama
      - groq
      - openai
    
    ollama:
      model: "llama3.2"
      base_url: "http://localhost:11434"
      enabled: true
      
    groq:
      model: "llama-3.1-70b-versatile"
      api_key_env: "GROQ_API_KEY"
      enabled: true
      
    openai:
      model: "gpt-4-turbo-preview"
      api_key_env: "OPENAI_API_KEY"
      enabled: false
  
  embedding:
    priority:
      - local
      - openai
    
    local:
      model: "all-MiniLM-L6-v2"
      enabled: true
      
    openai:
      model: "text-embedding-3-large"
      api_key_env: "OPENAI_API_KEY"
      enabled: false
```

---

## Phase 2: Intelligent Retrieval (Week 2)

### 2.1 Semantic Chunking

**Create:** `components/semantic_chunker.py`

```python
from typing import List, Dict, Any
from langchain.schema import Document
import re

class SemanticChunker:
    """Chunk documents based on semantic boundaries"""
    
    def __init__(self, min_chunk_size: int = 500, max_chunk_size: int = 2000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_document(self, document: Document) -> List[Document]:
        """Chunk document preserving semantic boundaries"""
        
        text = document.page_content
        metadata = document.metadata
        
        # Detect document structure
        structure = self._detect_structure(text)
        
        if structure['type'] == 'hierarchical':
            chunks = self._chunk_hierarchical(text, metadata)
        elif structure['type'] == 'list_based':
            chunks = self._chunk_list_based(text, metadata)
        else:
            chunks = self._chunk_paragraph_based(text, metadata)
        
        return chunks
    
    def _detect_structure(self, text: str) -> Dict[str, Any]:
        """Detect document structure type"""
        
        # Check for numbered sections (1., 1.1, Article 1, etc.)
        has_numbered_sections = bool(re.search(r'^\s*\d+\.', text, re.MULTILINE))
        has_articles = bool(re.search(r'Article\s+\d+', text, re.IGNORECASE))
        
        # Check for bullet points
        has_bullets = bool(re.search(r'^\s*[-•*]', text, re.MULTILINE))
        
        if has_numbered_sections or has_articles:
            return {'type': 'hierarchical', 'has_sections': True}
        elif has_bullets:
            return {'type': 'list_based'}
        else:
            return {'type': 'paragraph_based'}
    
    def _chunk_hierarchical(self, text: str, metadata: Dict) -> List[Document]:
        """Chunk hierarchical documents (regulations, legal docs)"""
        
        chunks = []
        
        # Split by major sections
        sections = re.split(r'\n(?=(?:Article|Section|Chapter)\s+\d+)', text)
        
        for section in sections:
            if not section.strip():
                continue
            
            # Extract section header
            header_match = re.match(r'((?:Article|Section|Chapter)\s+\d+[^\n]*)', section)
            header = header_match.group(1) if header_match else ""
            
            # If section is too large, split further
            if len(section) > self.max_chunk_size:
                subsections = self._split_large_section(section, header)
                chunks.extend(subsections)
            else:
                chunk_metadata = {
                    **metadata,
                    'section_header': header,
                    'chunk_type': 'section'
                }
                chunks.append(Document(page_content=section, metadata=chunk_metadata))
        
        return chunks
    
    def _split_large_section(self, section: str, header: str) -> List[Document]:
        """Split large sections while preserving context"""
        
        # Split by paragraphs
        paragraphs = section.split('\n\n')
        
        chunks = []
        current_chunk = header + "\n\n" if header else ""
        
        for para in paragraphs:
            if len(current_chunk) + len(para) > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunks.append(current_chunk)
                # Start new chunk with header for context
                current_chunk = f"[Continued from {header}]\n\n" if header else ""
            
            current_chunk += para + "\n\n"
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return [Document(page_content=c, metadata={'section_header': header}) 
                for c in chunks]
```

### 2.2 Multi-Stage Retrieval

**Create:** `retrieval/multi_stage.py`

```python
from typing import List, Dict, Any
import numpy as np

class MultiStageRetriever:
    """Multi-stage retrieval with re-ranking"""
    
    def __init__(self, vector_db, keyword_db, reranker=None):
        self.vector_db = vector_db
        self.keyword_db = keyword_db
        self.reranker = reranker
    
    def retrieve(self, query: str, top_k: int, use_reranking: bool = True) -> List[Dict]:
        """Multi-stage retrieval"""
        
        # Stage 1: Broad retrieval (get more candidates)
        candidates_k = top_k * 3
        
        # Vector search
        vector_results = self.vector_db.search(query, top_k=candidates_k)
        
        # Keyword search
        keyword_results = self.keyword_db.search(query, top_k=candidates_k)
        
        # Hybrid fusion (RRF)
        fused_results = self._reciprocal_rank_fusion(
            vector_results, keyword_results, k=60
        )
        
        # Stage 2: Re-ranking (if enabled)
        if use_reranking and self.reranker:
            reranked = self.reranker.rerank(query, fused_results[:top_k*2])
            final_results = reranked[:top_k]
        else:
            final_results = fused_results[:top_k]
        
        # Stage 3: Context expansion (get neighboring chunks)
        expanded_results = self._expand_context(final_results)
        
        return expanded_results
    
    def _reciprocal_rank_fusion(self, vector_results, keyword_results, k=60):
        """Combine results using RRF"""
        
        scores = {}
        
        for rank, result in enumerate(vector_results):
            idx = result['index']
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        
        for rank, result in enumerate(keyword_results):
            idx = result['index']
            scores[idx] = scores.get(idx, 0) + 1 / (k + rank)
        
        # Sort by score
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Build final results
        results = []
        for idx in sorted_indices:
            # Get metadata from vector results (or keyword)
            metadata = next((r for r in vector_results if r['index'] == idx), None)
            if not metadata:
                metadata = next((r for r in keyword_results if r['index'] == idx), None)
            
            if metadata:
                results.append({
                    **metadata,
                    'rrf_score': scores[idx]
                })
        
        return results
    
    def _expand_context(self, results: List[Dict]) -> List[Dict]:
        """Expand context by including neighboring chunks"""
        
        expanded = []
        
        for result in results:
            # Get neighboring chunks (previous and next)
            neighbors = self._get_neighbors(result)
            
            # Combine with original
            expanded_text = ""
            if neighbors['previous']:
                expanded_text += f"[Previous context]: {neighbors['previous']}\n\n"
            
            expanded_text += result['metadata']['text']
            
            if neighbors['next']:
                expanded_text += f"\n\n[Following context]: {neighbors['next']}"
            
            expanded_result = {
                **result,
                'expanded_text': expanded_text
            }
            expanded.append(expanded_result)
        
        return expanded
    
    def _get_neighbors(self, result: Dict) -> Dict[str, str]:
        """Get neighboring chunks from same document"""
        
        # This requires chunk_id metadata to track neighbors
        # Simplified version - implement based on your metadata structure
        
        return {
            'previous': None,  # Get from vector_db based on chunk_id - 1
            'next': None       # Get from vector_db based on chunk_id + 1
        }
```

---

## Phase 3: Adaptive Generation (Week 3)

### 3.1 Advanced Prompts with Chain-of-Thought

**Update:** `generation/prompts.py`

```python
class AdvancedPromptTemplate:
    """Advanced prompts with CoT and validation"""
    
    @staticmethod
    def rag_prompt_with_cot(context: str, question: str, query_type: str) -> str:
        """Generate RAG prompt with chain-of-thought reasoning"""
        
        if query_type == 'fact':
            return AdvancedPromptTemplate._fact_prompt(context, question)
        elif query_type == 'comparison':
            return AdvancedPromptTemplate._comparison_prompt(context, question)
        elif query_type == 'summary':
            return AdvancedPromptTemplate._summary_prompt(context, question)
        else:
            return AdvancedPromptTemplate._reasoning_prompt(context, question)
    
    @staticmethod
    def _fact_prompt(context: str, question: str) -> str:
        return f"""You are a precise information retrieval assistant. Answer the question using ONLY the provided context.

Context:
{context}

Question: {question}

Instructions:
1. Read the context carefully
2. Identify the specific information that answers the question
3. If the answer is in the context, provide it with the exact reference
4. If the answer is NOT in the context, say "I cannot find this information in the provided context"
5. Do NOT use external knowledge or make assumptions

Answer with citation:"""

    @staticmethod
    def _comparison_prompt(context: str, question: str) -> str:
        return f"""You are comparing information from regulatory documents. Use the provided context to make accurate comparisons.

Context:
{context}

Question: {question}

Instructions:
1. Identify the items being compared
2. Extract relevant information for EACH item from the context
3. Present the comparison in a structured format
4. Cite specific sections for each point
5. If information is missing for any item, explicitly state what's missing

Comparison:"""

    @staticmethod
    def _reasoning_prompt(context: str, question: str) -> str:
        return f"""You are analyzing complex regulatory information. Think step-by-step.

Context:
{context}

Question: {question}

Instructions:
1. Break down the question into sub-questions
2. For each sub-question, find relevant information in the context
3. Reason through the connections between different pieces of information
4. Synthesize a comprehensive answer
5. Cite all sources used in your reasoning

Step-by-step analysis:"""
```

### 3.2 Answer Validation

**Create:** `generation/validator.py`

```python
from typing import Dict, Any, List
import re

class AnswerValidator:
    """Validate generated answers for quality and grounding"""
    
    def __init__(self, llm_provider):
        self.llm = llm_provider
    
    def validate(self, question: str, answer: str, context: str) -> Dict[str, Any]:
        """Validate answer quality"""
        
        # Check grounding
        grounding_score = self._check_grounding(answer, context)
        
        # Check for hallucinations
        has_hallucination = self._detect_hallucination(answer, context)
        
        # Check completeness
        completeness = self._check_completeness(answer, question)
        
        # Calculate confidence
        confidence = self._calculate_confidence(
            grounding_score, has_hallucination, completeness
        )
        
        return {
            'is_valid': confidence > 0.6,
            'confidence': confidence,
            'grounding_score': grounding_score,
            'has_hallucination': has_hallucination,
            'completeness': completeness,
            'issues': self._identify_issues(grounding_score, has_hallucination, completeness)
        }
    
    def _check_grounding(self, answer: str, context: str) -> float:
        """Check if answer is grounded in context"""
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        grounded_claims = 0
        for claim in claims:
            if self._is_claim_in_context(claim, context):
                grounded_claims += 1
        
        if not claims:
            return 0.5  # Neutral if no claims
        
        return grounded_claims / len(claims)
    
    def _detect_hallucination(self, answer: str, context: str) -> bool:
        """Detect if answer contains hallucinated information"""
        
        # Check for common hallucination patterns
        hallucination_patterns = [
            r'based on (?:my|general) knowledge',
            r'I (?:believe|think|assume)',
            r'it is (?:likely|probably|possibly)',
            r'in my (?:opinion|view|experience)'
        ]
        
        for pattern in hallucination_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return True
        
        # Use LLM to verify (optional, more expensive)
        # verification = self._llm_verify_hallucination(answer, context)
        
        return False
    
    def _check_completeness(self, answer: str, question: str) -> float:
        """Check if answer addresses the question"""
        
        # Simple heuristic: check if answer is too short
        if len(answer.split()) < 10:
            return 0.3
        
        # Check if answer says "I don't know" or similar
        uncertain_patterns = [
            r'I (?:do not|don\'t) (?:have|know)',
            r'(?:cannot|can\'t) find',
            r'not (?:enough|sufficient) information',
            r'unclear|uncertain|not sure'
        ]
        
        for pattern in uncertain_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                return 0.4  # Low completeness if uncertain
        
        return 0.8  # Assume complete if no red flags
    
    def _calculate_confidence(self, grounding: float, has_hallucination: bool, 
                             completeness: float) -> float:
        """Calculate overall confidence score"""
        
        if has_hallucination:
            return 0.2  # Very low confidence if hallucinating
        
        # Weighted average
        confidence = (grounding * 0.5) + (completeness * 0.5)
        
        return confidence
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from answer"""
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', answer)
        
        # Filter out non-factual sentences
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and not sentence.startswith(('I ', 'You ', 'We ')):
                claims.append(sentence)
        
        return claims
    
    def _is_claim_in_context(self, claim: str, context: str) -> bool:
        """Check if claim is supported by context"""
        
        # Simple substring matching (can be improved with semantic similarity)
        claim_words = set(claim.lower().split())
        context_words = set(context.lower().split())
        
        # Check overlap
        overlap = len(claim_words & context_words) / len(claim_words)
        
        return overlap > 0.5  # At least 50% word overlap
    
    def _identify_issues(self, grounding: float, has_hallucination: bool, 
                        completeness: float) -> List[str]:
        """Identify specific issues with the answer"""
        
        issues = []
        
        if grounding < 0.5:
            issues.append("Low grounding in context")
        
        if has_hallucination:
            issues.append("Contains hallucinated information")
        
        if completeness < 0.5:
            issues.append("Incomplete answer")
        
        return issues
```

---

## Phase 4-6: Detailed Plans

Due to length constraints, I'll provide the high-level structure. The implementation follows similar patterns:

### Phase 4: Learning & Optimization
- Feedback collection system
- Parameter optimization based on feedback
- A/B testing framework
- Performance monitoring

### Phase 5: Production Readiness
- Redis caching for embeddings and queries
- Async processing with asyncio
- FastAPI endpoints
- Authentication (JWT)
- Rate limiting

### Phase 6: Testing & Refinement
- Comprehensive test suite
- Evaluation pipeline
- Documentation
- Deployment configs (Docker, K8s)

---

## Migration Path: Local → Paid APIs

### Easy Migration Design

**Current (Local):**
```python
# config/providers.yaml
providers:
  llm:
    priority: [ollama]  # Only local
```

**After Testing (Hybrid):**
```python
providers:
  llm:
    priority: [ollama, groq]  # Fallback to Groq if local fails
```

**Production (Paid):**
```python
providers:
  llm:
    priority: [groq, openai]  # Paid APIs only
```

**No code changes needed!** Just update config.

---

## Success Metrics

### Week 1-2:
- [ ] Architecture implemented
- [ ] 5 strategies working
- [ ] Provider abstraction complete
- [ ] Accuracy: 60-65%

### Week 3-4:
- [ ] Advanced prompts
- [ ] Answer validation
- [ ] Multi-stage retrieval
- [ ] Accuracy: 70-75%

### Week 5-6:
- [ ] Feedback loops
- [ ] Production features
- [ ] Full evaluation
- [ ] Accuracy: 75-80%

---

## Next Steps

1. Review this plan
2. Start with Phase 1.1 (base architecture)
3. Implement incrementally
4. Test each component
5. Iterate based on results

This plan gives you a **production-ready, adaptive system** that works with local models and easily scales to paid APIs.

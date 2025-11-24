# Adaptive RAG System - Implementation Details

## Architecture Overview

### Core Components
- **Query Analyzer** (`core/query_analyzer.py`): Classifies queries using LLM with heuristic fallback
- **Strategy Selector** (`core/strategy_selector.py`): Routes queries to optimal strategies
- **Metrics Tracker** (`core/metrics.py`): Tracks performance and usage

### Provider Abstraction
- **Ollama** (`providers/ollama_provider.py`): Local LLM (free)
- **Groq** (`providers/groq_provider.py`): Fast cloud API
- **OpenAI** (`providers/openai_provider.py`): GPT-4 and other models
- **Local Embeddings** (`providers/local_embedding_provider.py`): sentence-transformers

### Strategies

| Strategy | Query Type | top_k | Chunk Size | Temperature |
|----------|-----------|-------|------------|-------------|
| Simple Fact | fact (simple) | 3 | 800 | 0.0 |
| Complex Fact | fact (complex) | 10 | 1200 | 0.1 |
| Summary | summary | 30 | 2000 | 0.2 |
| Comparison | comparison | 20 | 1500 | 0.1 |
| Reasoning | reasoning | 15 | 1800 | 0.15 |

## How It Works

1. **Query Analysis**: System analyzes question and classifies type/complexity
2. **Strategy Selection**: Routes to optimal strategy based on classification
3. **Execution**: Strategy retrieves documents and generates answer
4. **Metrics**: Tracks performance for monitoring

## Configuration

### Provider Priority (`config/providers.yaml`)
```yaml
providers:
  llm:
    priority: [ollama, groq, openai]  # Try in order
```

### Easy API Migration
- **Local**: `priority: [ollama]`
- **Hybrid**: `priority: [ollama, groq]`
- **Production**: `priority: [groq, openai]`

Just edit YAML - no code changes needed!

## Expected Performance

- **Overall Accuracy**: 46% → 65-75% (+19-29%)
- **Fact Retrieval**: 56% → 70-80%
- **Context Understanding**: 35% → 55-65%
- **Summarization**: 40-46% → 60-70%

## Usage

```python
from adaptive_rag import AdaptiveRAGPipeline

# Initialize
rag = AdaptiveRAGPipeline()

# Query
result = rag.query("What is the maximum fuel flow rate?")

# Result includes:
# - answer: Generated answer
# - strategy: Which strategy was used
# - query_profile: Query classification
# - sources: Retrieved documents
# - response_time: Time taken
```

## Testing

```bash
# Quick test
python test_adaptive.py

# Full evaluation
python evaluation/evaluate_adaptive.py
```

## Docker

```bash
# Build and run
docker-compose build
docker-compose up -d

# Test
docker-compose exec rag-app python test_adaptive.py
```

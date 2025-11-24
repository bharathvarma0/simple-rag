# Adaptive RAG System

A production-ready, adaptive Retrieval-Augmented Generation (RAG) system with intelligent query routing, YAML-based configuration, and provider abstraction for seamless migration between local and cloud LLMs.

## 🎯 Key Features

- **Adaptive Query Routing**: Automatically classifies queries and routes to optimal strategies
- **5 Specialized Strategies**: Fact retrieval, summarization, comparison, reasoning - each with tuned parameters
- **YAML Configuration**: Industry-standard config management (no code changes to adjust parameters)
- **Provider Abstraction**: Easy switching between Ollama (local), Groq, and OpenAI
- **Hybrid Approach**: YAML configs with hardcoded fallbacks for reliability
- **Performance Tracking**: Built-in metrics and monitoring

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Ollama (for local LLM) OR Groq/OpenAI API key
- Your documents in `data/pdfs/`

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Build vector store (one-time setup)
python -c "from data_ingestion.ingest import ingest_documents; from augmentation.vector_db import VectorDatabase; docs, chunks = ingest_documents(); vdb = VectorDatabase(); vdb.build_from_documents(docs)"
```

### Usage

#### Quick Test
```bash
python test_adaptive.py
```

#### Full Evaluation
```bash
python evaluation/evaluate_adaptive.py
```

#### Use in Code
```python
from adaptive_rag import AdaptiveRAGPipeline

# Initialize
rag = AdaptiveRAGPipeline()

# Query (automatic strategy selection)
result = rag.query("What is the maximum fuel flow rate?")

print(result['answer'])
print(f"Strategy: {result['strategy']}")
print(f"Query type: {result['query_profile']['query_type']}")
```

## 📊 How It Works

```
User Question
     ↓
Query Analyzer (classifies: fact/summary/comparison/reasoning)
     ↓
Strategy Selector (routes to optimal strategy)
     ↓
Strategy Execution (loads params from YAML, retrieves documents)
     ↓
Result + Metrics
```

### Strategies

| Strategy | Query Type | top_k | Use Case |
|----------|-----------|-------|----------|
| **Simple Fact** | fact (simple) | 3 | "What is X?" |
| **Complex Fact** | fact (complex) | 10 | "What are all details about X?" |
| **Summary** | summary | 30 | "Summarize X" |
| **Comparison** | comparison | 20 | "Compare X and Y" |
| **Reasoning** | reasoning | 15 | "How does X affect Y?" |

## ⚙️ Configuration

### Provider Configuration (`config/providers.yaml`)

Switch LLM providers without code changes:

```yaml
providers:
  llm:
    priority: [ollama, groq, openai]  # Try in order
```

**Local Development:**
```yaml
priority: [ollama]
```

**Production:**
```yaml
priority: [groq, openai]
```

### Strategy Configuration (`config/strategies.yaml`)

Tune strategy parameters without code changes:

```yaml
simple_fact:
  top_k: 3
  temperature: 0.0
  max_tokens: 500

summary:
  top_k: 30
  temperature: 0.2
  max_tokens: 2048
```

**To change parameters**: Just edit the YAML file - no redeployment needed!

### Environment Variables (`.env`)

```bash
GROQ_API_KEY=your_groq_key_here
OPENAI_API_KEY=your_openai_key_here
OLLAMA_BASE_URL=http://localhost:11434
```

## 🐳 Docker Deployment

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Test
docker-compose exec rag-app python test_adaptive.py

# Evaluate
docker-compose exec rag-app python evaluation/evaluate_adaptive.py
```

## 📁 Project Structure

```
rag-project/
├── core/                    # Core architecture
│   ├── query_analyzer.py    # Query classification
│   ├── strategy_selector.py # Strategy routing
│   └── metrics.py           # Performance tracking
├── providers/               # LLM provider abstraction
│   ├── ollama_provider.py   # Local Ollama
│   ├── groq_provider.py     # Groq API
│   └── openai_provider.py   # OpenAI API
├── strategies/              # Specialized strategies (YAML-configured)
│   ├── fact_strategy.py     # Fact retrieval
│   ├── summary_strategy.py  # Summarization
│   ├── comparison_strategy.py # Comparison
│   └── reasoning_strategy.py  # Reasoning
├── config/                  # Configuration
│   ├── providers.yaml       # Provider selection
│   └── strategies.yaml      # Strategy parameters
├── adaptive_rag.py          # Main pipeline
├── test_adaptive.py         # Quick test
└── evaluation/              # Evaluation scripts
    ├── questions.py         # Test questions
    └── evaluate_adaptive.py # Evaluation runner
```

## 📈 Performance

Expected accuracy improvements over v1:
- **Overall**: 46% → 65-75% (+19-29%)
- **Fact Retrieval**: 56% → 70-80%
- **Context Understanding**: 35% → 55-65%
- **Summarization**: 40-46% → 60-70%

## 🔧 Customization

### Add New Strategy
1. Create new strategy in `strategies/`
2. Inherit from `BaseStrategy`
3. Implement `get_params()` and `_generate_prompt()`
4. Add to `config/strategies.yaml`
5. Register in `adaptive_rag.py`

### Add New Provider
1. Create provider in `providers/`
2. Inherit from `BaseLLMProvider`
3. Implement `generate()` and `is_available()`
4. Add to `config/providers.yaml`

### Tune Parameters
Edit `config/strategies.yaml`:
```yaml
simple_fact:
  top_k: 5  # Change from 3 to 5
  temperature: 0.1  # Adjust temperature
```

## 📚 Documentation

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - Detailed implementation plan and architecture
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Quick technical overview

## 🛠️ Development

### Run Tests
```bash
python test_adaptive.py
```

### Run Evaluation
```bash
python evaluation/evaluate_adaptive.py
```

### Check Logs
Logs are output to console with detailed information about query classification, strategy selection, and retrieval.

## 🐛 Troubleshooting

### Ollama not available
- Ensure Ollama is running: `ollama serve`
- Check connection: `curl http://localhost:11434/api/tags`

### No provider available
- Set API key: `export GROQ_API_KEY=your_key`
- Or edit `config/providers.yaml` to enable provider

### Low accuracy
- Increase `top_k` in `config/strategies.yaml`
- Adjust temperature for more/less creative responses
- Check document quality in `data/pdfs/`

### Configuration not loading
- Check `config/strategies.yaml` syntax (valid YAML)
- System will fallback to hardcoded defaults if YAML fails
- Check logs for "Loaded strategy configuration" message

## 📝 License

See [LICENSE](LICENSE) file.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Test with `python test_adaptive.py`
5. Submit pull request

---

**Built with ❤️ using adaptive RAG architecture and industry-standard YAML configuration**

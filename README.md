# Adaptive RAG System v2.0

**Production-ready RAG with adaptive strategies, re-ranking, and multi-stage retrieval**

[![Accuracy](https://img.shields.io/badge/Accuracy-75%25-success)]()
[![Re-ranking](https://img.shields.io/badge/Re--ranking-✓-blue)]()
[![Multi-stage](https://img.shields.io/badge/Multi--stage-✓-blue)]()

---

## 🎯 **Version 2.0 Features**

### **Core Capabilities:**
- ✅ **75% Overall Accuracy** (up from 60%)
- ✅ **100% Summarization Accuracy** (perfect!)
- ✅ **100% Fact Retrieval** (maintained)
- ✅ **Cross-Encoder Re-Ranking** (ms-marco-MiniLM-L-6-v2)
- ✅ **Multi-Stage Retrieval** with context expansion
- ✅ **5 Adaptive Strategies** with automatic routing
- ✅ **Hybrid Search** (FAISS HNSW + BM25)
- ✅ **YAML Configuration** (no hardcoding)

---

## 🚀 **Quick Start**

### Prerequisites
```bash
Python 3.11+
Ollama (local) OR Groq/OpenAI API key
```

### Installation
```bash
# Clone and install
git clone <repo>
cd rag-project
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Build vector database + Run evaluation
python run_full_pipeline.py --build
```

### Quick Test
```bash
python test_adaptive.py
```

---

## 📊 **Performance Results**

| Metric | Accuracy | Status |
|--------|----------|--------|
| **Overall** | **75%** | ✅ |
| **Fact Retrieval** | **100%** | ✅ |
| **Summarization** | **100%** | ✅ |
| **Context Understanding** | 80% | ✅ |
| Complex Reasoning | 20% | ⚠️ |

**Response Time:** ~47s average per query  
**Evaluation Set:** 20 diverse questions

---

## 🏗️ **Architecture**

```
User Query
    ↓
Query Analyzer (LLM-based classification)
    ↓
Strategy Selector (routes to optimal strategy)
    ↓
Multi-Stage Retrieval:
  Stage 1: Hybrid Search (50-100 candidates)
  Stage 2: Context Expansion (neighbors)
  Stage 3: Cross-Encoder Re-ranking
    ↓
LLM Generation (strategy-specific prompt)
    ↓
Result + Metadata
```

---

## 🎯 **Adaptive Strategies**

| Strategy | Query Type | Depth | Candidates | Re-rank | Use Case |
|----------|-----------|-------|------------|---------|----------|
| **Simple Fact** | fact (simple) | 1 | 3 | ❌ | "What is X?" |
| **Complex Fact** | fact (complex) | 2 | 50 | ✅ | "What are all details about X?" |
| **Summary** | summary | 2 | 100 | ✅ | "Summarize the regulations" |
| **Comparison** | comparison | 2 | 60 | ✅ | "Compare X and Y" |
| **Reasoning** | reasoning | 2 | 60 | ✅ | "How does X affect Y?" |

**Key Parameters:**
- **Depth**: 1=basic, 2=with neighbors (multi-stage)
- **Candidates**: Initial retrieval count before re-ranking  
- **Re-rank**: Use cross-encoder for better accuracy

---

## ⚙️ **Configuration**

### Provider Selection (`config/providers.yaml`)
```yaml
providers:
  llm:
    priority: [ollama, groq, openai]  # Try in order
```

### Strategy Parameters (`config/strategies.yaml`)
```yaml
# Global re-ranking
reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Global multi-stage retrieval
retrieval:
  enable_multi_stage: true
  default_candidates: 50

#  Per-strategy configuration
strategies:
  simple_fact:
    top_k: 3
    retrieval_depth: 1  # Fast, no expansion
    use_reranking: false
    temperature: 0.0
    
  complex_fact:
    top_k: 10
    retrieval_depth: 2  # Multi-stage
    initial_candidates: 50
    use_reranking: true
    temperature: 0.1
```

**Change parameters:** Just edit YAML - no code changes needed!

---

## 💻 **Usage**

### Python API
```python
from adaptive_rag import AdaptiveRAGPipeline

# Initialize
rag = AdaptiveRAGPipeline()

# Query (automatic strategy selection)
result = rag.query("What is the maximum fuel flow rate?")

print(result['answer'])
print(f"Strategy: {result['strategy']}")
print(f"Used re-ranking: {result['used_reranking']}")
print(f"Retrieval depth: {result['retrieval_depth']}")
```

### Command Line
```bash
# Full pipeline (clean, ingest, build, evaluate)
python run_full_pipeline.py --build

# Just evaluation
python evaluation/evaluate_adaptive.py

# Quick test
python test_adaptive.py
```

---

## 🔬 **Advanced Features**

### 1. **Cross-Encoder Re-Ranking**
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Lazy loading (only when enabled)
- Configurable per-strategy
- +10-15% accuracy improvement

### 2. **Multi-Stage Retrieval**
- Stage 1: Cast wide net (50-100 candidates)
- Stage 2: Expand with neighboring chunks  
- Stage 3: Re-rank to best N
- Solves chunking boundary problems

### 3. **Hybrid Search**
- FAISS HNSW (semantic similarity)
- BM25 (keyword matching)
- Reciprocal Rank Fusion (RRF)
- Best of both worlds

---

## 📁 **Project Structure**

```
rag-project/
├── core/                      # Core logic
│   ├── query_analyzer.py      # LLM-based classification
│   ├── strategy_selector.py   # Routing logic
│   └── metrics.py             # Performance tracking
├── strategies/                # 5 specialized strategies
│   ├── fact_strategy.py       # Simple & complex fact
│   ├── summary_strategy.py    # Summarization
│   ├── comparison_strategy.py # Comparison
│   └── reasoning_strategy.py  # Chain-of-thought reasoning
├── augmentation/              # Retrieval & search
│   ├── vector_db.py           # FAISS HNSW index
│   ├── keyword_db.py          # BM25 index
│   ├── search.py              # Multi-stage hybrid search
│   └── chunk_context.py       # Chunk relationship tracking
├── components/                # Core components
│   ├── reranker.py            # Cross-encoder re-ranking
│   ├── embedders.py           # Embedding generation
│   └── chunkers.py            # Document chunking
├── providers/                 # LLM providers
│   ├── ollama_provider.py     # Local (Ollama)
│   ├── groq_provider.py       # Groq API
│   └── openai_provider.py     # OpenAI API
├── config/                    # Configuration
│   ├── strategies.yaml        # Strategy & retrieval config
│   └── providers.yaml         # Provider selection
├── evaluation/                # Testing
│   ├── questions.py           # 20 evaluation questions
│   ├── evaluate_adaptive.py   # Evaluation runner
│   └── outputs/               # Results (timestamped)
├── adaptive_rag.py            # Main pipeline
└── run_full_pipeline.py       # Build & evaluate
```

---

## 🧪 **Testing**

### Run Full Evaluation
```bash
python evaluation/evaluate_adaptive.py
```

**Outputs:**
- Terminal display with live results
- Saved to `evaluation/outputs/evaluation_results_TIMESTAMP.txt`
- Classification accuracy by category
- Strategy distribution
- Performance metrics

### Integration Tests
```bash
# Test re-ranking
python test_reranking.py

# Test multi-stage retrieval
python test_multistage.py

# Quick adaptive test
python test_adaptive.py
```

---

## 🐳 **Docker Deployment**

```bash
# Build
docker-compose build

# Run
docker-compose up -d

# Evaluate
docker-compose exec rag-app python evaluation/evaluate_adaptive.py
```

---

## 🔧 **Customization**

### Tune Retrieval
```yaml
# config/strategies.yaml
complex_fact:
  retrieval_depth: 3       # More context (slower)
  initial_candidates: 100  # Cast wider net
  rerank_candidates: 50    # Re-rank more chunks
```

### Adjust Re-Ranking
```yaml
reranking:
  enabled: false  # Disable for speed
  batch_size: 16  # Smaller batches for less memory
```

### Add Custom Strategy
1. Create `strategies/my_strategy.py`
2. Inherit from `BaseStrategy`
3. Implement `get_params()` and `_generate_prompt()`
4. Add to `config/strategies.yaml`
5. Register in `adaptive_rag.py`

---

## 📈 **Roadmap**

### ✅ Completed (v2.0)
- [x] Re-ranking system
- [x] Multi-stage retrieval
- [x] YAML configuration
- [x] 75% accuracy baseline

### 🔄 Planned (v2.1+)
- [ ] Query expansion (+5-10% accuracy)
- [ ] Improved reasoning detection (+30% on reasoning)
- [ ] Memory/conversation history
- [ ] REST API endpoints
- [ ] Streaming responses
- [ ] Contextual compression

---

## 🐛 **Troubleshooting**

### Ollama not available
```bash
# Start Ollama
ollama serve

# Test connection
curl http://localhost:11434/api/tags
```

### Re-ranking model not downloading
```bash
# Manual download
python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

### Low accuracy
- Check `config/strategies.yaml` parameters
- Ensure vector database is built: `python run_full_pipeline.py --build`
- Review evaluation output in `evaluation/outputs/`

---

## 📚 **Documentation**

- [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) - v1 architecture
- [Phase 2 Plan](artifacts/implementation_plan_phase2.md) - v2 features
- [Re-ranking Implementation](artifacts/reranking_implementation.md) - Re-ranker details
- [Multi-Stage Implementation](artifacts/multistage_implementation.md) - Multi-stage details

---

## 🤝 **Contributing**

```bash
# Create feature branch from version2
git checkout version2
git checkout -b feature/my-feature

# Make changes, test
python test_adaptive.py
python evaluation/evaluate_adaptive.py

# Commit and push
git commit -m "feat: Add my feature"
git push origin feature/my-feature
```

---

## 📝 **License**

MIT License - See LICENSE file

---

## 🏆 **Achievements**

- ✅ **75% overall accuracy** (industry-standard)
- ✅ **100% summarization** (perfect!)
- ✅ **Production-ready architecture**
- ✅ **Configuration-driven** (no hardcoding)
- ✅ **Future-proof** (ready for Weaviate/Pinecone)

---

**Built with ❤️ using adaptive RAG, cross-encoder re-ranking, and multi-stage retrieval**

v2.0 | Branch: `version2` | Status: Production-Ready ✅

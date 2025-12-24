# Traditional RAG System

A clean, modular implementation of a Retrieval-Augmented Generation (RAG) system for document question-answering.

## 📁 Project Structure

```
RAG-Tutorials/
├── components/              # Reusable components
│   ├── __init__.py
│   ├── loaders.py          # Document loaders (PDF, TXT, CSV, etc.)
│   ├── chunkers.py         # Text chunking utilities
│   └── embedders.py        # Embedding generation
├── data/                   # Data storage
│   └── pdfs/              # Place your PDF files here
├── data_ingestion/         # Data ingestion scripts
│   ├── __init__.py
│   └── ingest.py          # Main ingestion script
├── augmentation/          # Vector database and similarity search
│   ├── __init__.py
│   ├── vector_db.py      # Qdrant vector database management
│   └── search.py         # Similarity search utilities
├── generation/            # LLM generation
│   ├── __init__.py
│   └── rag.py            # RAG pipeline (retrieval + generation)
├── vector_store/         # Generated vector store (created automatically)
├── api/                  # API endpoints
│   ├── main.py           # FastAPI application entry point
│   └── routes.py         # API routes
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🚀 Getting Started

### 1. Installation

#### Local Installation
```bash
# Install dependencies
pip install -r requirements.txt
```

#### Docker Installation
```bash
# Build the image
docker build -t rag-tutorial .

# Run the container
docker run -it --env-file .env -v $(pwd)/data:/app/data -v $(pwd)/vector_store:/app/vector_store rag-tutorial
```

### 2. Set Environment Variables

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

Get your API key from: https://platform.openai.com/

### 3. Add Your Documents

Place your PDF files (or other supported formats) in the `data/pdfs/` directory:

```bash
mkdir -p data/pdfs
# Copy your PDF files to data/pdfs/
```

### 4. Build Vector Store

```bash
# Build vector store from documents
python app.py --build --data-dir data/pdfs
```

### 5. Query the System

```bash
# Query with a question
python app.py --query "What is attention mechanism?"

# Interactive mode (default)
python app.py
```

## 📖 Usage

### API Usage

The system exposes a FastAPI interface.

```bash
# Start the API
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

### Python API

```python
from generation.rag import RAGPipeline

# Initialize RAG pipeline
rag = RAGPipeline(persist_dir="vector_store")

# Query the system
result = rag.query("What is your question?", top_k=5)

# Get answer
answer = result["answer"]
sources = result["sources"]
context = result["context"]

# Simple query (just answer)
answer = rag.ask("What is your question?")
```

### Build Vector Store Programmatically

```python
from data_ingestion.ingest import ingest_documents
from augmentation.vector_db import VectorDatabase

# Ingest documents
raw_docs, chunks = ingest_documents("data/pdfs")

# Build vector database
vector_db = VectorDatabase(persist_dir="vector_store")
vector_db.build_from_documents(raw_docs)
```

## 🧩 Components

### Document Loaders (`components/loaders.py`)
- Supports: PDF, TXT, CSV, Excel (.xlsx), Word (.docx), JSON
- Automatically finds and loads all supported files from directory

### Text Chunkers (`components/chunkers.py`)
- Splits documents into smaller chunks
- Configurable chunk size and overlap

### Embedding Generators (`components/embedders.py`)
- Uses OpenAI Embeddings
- Model: `text-embedding-3-small` (1536 dimensions)

### Vector Database (`augmentation/vector_db.py`)
- Qdrant-based vector storage
- Persistent storage to disk
- Automatic loading and saving

### Similarity Search (`augmentation/search.py`)
- Semantic search in vector database
- Returns top-k similar documents
- Configurable similarity thresholds

### RAG Pipeline (`generation/rag.py`)
- Traditional RAG: Retrieve → Generate
- Uses OpenAI LLM for generation
- Returns answers with source citations

## ⚙️ Configuration

### Vector Store Settings

Edit `augmentation/vector_db.py`:
- `persist_dir`: Directory for vector store
- `embedding_model`: Embedding model name

### Chunking Settings

Edit `data_ingestion/ingest.py`:
- `chunk_size`: Size of text chunks (default: 1000)
- `chunk_overlap`: Overlap between chunks (default: 200)

### LLM Settings

Edit `generation/rag.py`:
- `llm_model`: OpenAI model name (default: `gpt-4o-mini`)
- `temperature`: LLM temperature (default: 0.1)
- `max_tokens`: Maximum tokens (default: 1024)

## 📝 Supported File Formats

- **PDF** (`.pdf`) - Using PyPDFLoader
- **Text** (`.txt`) - Using TextLoader
- **CSV** (`.csv`) - Using CSVLoader
- **Excel** (`.xlsx`) - Using UnstructuredExcelLoader
- **Word** (`.docx`) - Using Docx2txtLoader
- **JSON** (`.json`) - Using JSONLoader

## 🔧 Dependencies

- `langchain` - Document processing framework
- `langchain-openai` - OpenAI integration
- `openai` - OpenAI SDK
- `qdrant-client` - Vector database
- `pypdf` / `pymupdf` - PDF processing

## 📋 Workflow

1. **Data Ingestion**: Load documents from `data/pdfs/`
2. **Chunking**: Split documents into smaller pieces
3. **Embedding**: Generate vector embeddings
4. **Vector Storage**: Store in FAISS database
5. **Query**: User asks a question
6. **Retrieval**: Find similar documents
7. **Generation**: LLM generates answer from context
8. **Response**: Return answer with sources

## 🐛 Troubleshooting

### Vector Store Not Found
```bash
# Build the vector store first
python app.py --build
```

### OPENAI_API_KEY Not Found
```bash
# Create .env file with your API key
echo "OPENAI_API_KEY=your_key_here" > .env
```

### No Documents Found
```bash
# Make sure PDFs are in data/pdfs/
ls data/pdfs/
```

## 📄 License

GNU General Public License v3.0

## 🤝 Contributing

This is a clean, modular RAG implementation. Feel free to extend and modify for your needs!

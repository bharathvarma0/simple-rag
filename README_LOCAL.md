# Local RAG Setup Guide

This document details the changes made to enable a fully local, privacy-focused RAG system using Ollama and Docker.

## 🔄 Changes Made

### 1. Architecture
- **Local LLM**: Switched from cloud APIs (Groq/OpenAI) to **Ollama** (running locally).
- **Local Embeddings**: Switched to `all-MiniLM-L6-v2` for fast, local embedding generation.
- **Vector Database**: Persisted locally on disk using FAISS.

### 2. Components Added
- **`Dockerfile`**: Lightweight Python 3.11 image for consistent execution.
- **`docker-compose.yml`**: Orchestration for running the app with volume persistence and host networking.
- **`tests/`**: Automated tests to verify local configuration.

### 3. Code Updates
- **`config/settings.py`**: Updated defaults for Ollama (`llama3.2`) and local paths.
- **`generation/llm_wrapper.py`**: Added conditional support for `ChatOllama` to avoid dependency crashes.
- **`requirements.txt`**: Added `langchain-community`, `langchain-text-splitters`, and `langchain-openai`.

---

## 🚀 How to Run

### Prerequisites
1.  **Install Ollama**: [Download here](https://ollama.com/).
2.  **Pull Model**:
    ```bash
    ollama pull llama3.2
    ```

### Option 1: Run with Docker (Recommended)
This method ensures it works exactly the same on Windows and macOS.

1.  **Start the App**:
    ```bash
    docker-compose up --build
    ```
2.  **Enter the Container** (in a new terminal):
    ```bash
    docker-compose exec rag-app python app.py
    ```

### Option 2: Run Locally (Python)

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the App**:
    ```bash
    python app.py
    ```

---

## 🛠️ Workflow

1.  **First Time Only**:
    If you add **new PDF files** to `data/pdfs/`, you must rebuild the database:
    ```bash
    python app.py --build
    ```

2.  **Daily Use**:
    Just run the app and ask questions:
    ```bash
    python app.py --query "Your question here"
    ```

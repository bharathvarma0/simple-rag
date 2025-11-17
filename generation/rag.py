"""
RAG pipeline: Retrieval-Augmented Generation
Combines retrieval with LLM generation
"""

from typing import Dict, Any, Optional
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))

from augmentation.vector_db import VectorDatabase
from augmentation.search import SimilaritySearch
from generation.llm_wrapper import LLMWrapper
from generation.prompts import PromptTemplate
from utils.logger import get_logger
from config import get_settings

logger = get_logger(__name__)


class RAGPipeline:
    """Traditional RAG pipeline combining retrieval and generation"""
    
    def __init__(
        self, 
        persist_dir: Optional[str] = None,
        embedding_model: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            persist_dir: Directory containing vector store (uses config default if None)
            embedding_model: Embedding model name (uses config default if None)
            llm_provider: LLM provider (uses config default if None)
            llm_model: LLM model name (uses config default if None)
            llm_api_key: LLM API key (uses env var if None)
            llm_temperature: LLM temperature (uses config default if None)
            llm_max_tokens: LLM max tokens (uses config default if None)
        """
        self.settings = get_settings()
        
        # Use provided values or defaults from config
        self.persist_dir = persist_dir or self.settings.vector_store.persist_dir
        self.embedding_model = embedding_model or self.settings.embedding.model_name
        
        # Initialize vector database
        self.vector_db = VectorDatabase(
            persist_dir=self.persist_dir,
            embedding_model=self.embedding_model
        )
        
        # Load vector database
        if self.vector_db.exists():
            self.vector_db.load()
        else:
            raise FileNotFoundError(
                f"Vector database not found at {self.persist_dir}. "
                "Please run data ingestion and build the vector database first."
            )
        
        # Initialize search
        self.search = SimilaritySearch(self.vector_db)
        
        # Initialize LLM wrapper
        self.llm = LLMWrapper(
            provider=llm_provider,
            model_name=llm_model,
            api_key=llm_api_key,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )
        
        # Initialize prompt template
        self.prompt_template = PromptTemplate()
        
        logger.info(f"RAG pipeline initialized with LLM: {self.llm.model_name}")
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User question
            top_k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        top_k = top_k or self.settings.retrieval.top_k
        
        logger.info(f"Processing question: {question}")
        
        # Retrieve relevant documents
        results = self.search.search(question, top_k=top_k)
        
        if not results:
            logger.warning("No relevant documents found")
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "context": "",
                "num_sources": 0
            }
        
        # Build context from retrieved documents
        context = self.search.get_context(question, top_k=top_k)
        
        # Generate prompt using template
        prompt = self.prompt_template.rag_prompt(context=context, question=question)
        
        # Generate answer using LLM
        answer = self.llm.generate(prompt)
        
        # Extract source information
        sources = []
        for i, result in enumerate(results, 1):
            source_info = {
                "rank": i,
                "similarity_score": result["similarity_score"],
                "distance": result["distance"],
                "source": result["metadata"].get("source", "unknown"),
                "preview": result["metadata"].get("text", "")[:200] + "..."
            }
            sources.append(source_info)
        
        logger.info(f"Generated answer with {len(sources)} sources")
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "num_sources": len(sources)
        }
    
    def ask(self, question: str, top_k: Optional[int] = None) -> str:
        """
        Simple query method that returns only the answer
        
        Args:
            question: User question
            top_k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            Answer string
        """
        result = self.query(question, top_k=top_k)
        return result["answer"]


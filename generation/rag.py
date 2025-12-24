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
        
        # Initialize memory
        from generation.memory import ConversationMemory
        self.memory = ConversationMemory()
        
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
        
        # Rewrite question if history exists
        search_query = question
        history = self.memory.get_history_string()
        
        if history and self.settings.memory.enabled:
            logger.info("Rewriting question based on history...")
            rewrite_prompt = self.prompt_template.rewrite_prompt(history, question)
            search_query = self.llm.generate(rewrite_prompt).strip()
            logger.info(f"Rewritten query: '{search_query}'")
        
        # Retrieve relevant documents using search_query
        # We request more than needed to allow for dynamic filtering
        results = self.search.search(search_query, top_k=top_k * 2)
        
        if not results:
            logger.warning("No relevant documents found")
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "context": "",
                "num_sources": 0,
                "rewritten_query": None
            }
        
        # Dynamic Context Selection
        # Select chunks based on token budget and score threshold
        final_results = []
        current_tokens = 0
        max_context_tokens = 1500 # Token budget for context
        min_score_threshold = 0.3 # Minimum similarity score
        
        # Estimate tokens (approx 4 chars per token)
        def estimate_tokens(text):
            return len(text) / 4
            
        for res in results:
            # Check score threshold
            if res["similarity_score"] < min_score_threshold:
                continue
                
            text = res["metadata"].get("text", "")
            tokens = estimate_tokens(text)
            
            if current_tokens + tokens > max_context_tokens:
                break
                
            final_results.append(res)
            current_tokens += tokens
            
            # Stop if we have enough high-quality chunks (e.g., 5)
            if len(final_results) >= top_k:
                break
        
        # If no chunks passed the threshold, fallback to top 1 (if available)
        if not final_results and results:
            final_results = [results[0]]
            
        # Build context from selected documents
        texts = [r["metadata"].get("text", "") for r in final_results]
        context = "\n\n".join(texts)
        
        # Generate prompt using template
        prompt = self.prompt_template.rag_prompt(context=context, question=question)
        
        # Generate answer using LLM
        answer = self.llm.generate(prompt)
        
        # Update memory
        if self.settings.memory.enabled:
            self.memory.add_turn(question, answer)
            # Periodically summarize memory (e.g., every turn, let the memory class decide based on size)
            self.memory.summarize(self.llm)
        
        # Extract source information
        sources = []
        for i, result in enumerate(final_results, 1):
            source_info = {
                "rank": i,
                "distance": result["similarity_score"],
                "source": result["metadata"].get("source", "unknown"),
                "preview": result["metadata"].get("text", "")[:300]
            }
            sources.append(source_info)
        
        logger.info(f"Generated answer with {len(sources)} sources")
        
        return {
            "answer": answer,
            "sources": sources,
            "context": context,
            "num_sources": len(sources),
            "rewritten_query": search_query if search_query != question else None
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


"""
Application settings and configuration
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384  # Will be set automatically based on model
    
    
@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = field(default_factory=lambda: ["\n\n", "\n", " ", ""])


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_dir: str = "vector_store"
    index_type: str = "hnsw"  # Options: "flat", "hnsw"
    hnsw_m: int = 32  # Number of connections per node for HNSW


@dataclass
class LLMConfig:
    """Configuration for LLM"""
    provider: str = "openai"  # groq, openai, etc.
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 1024
    api_key: Optional[str] = None
    
    def __post_init__(self):
        """Load API key from environment if not provided"""
        if self.api_key is None:
            if self.provider == "groq":
                self.api_key = os.getenv("GROQ_API_KEY")
            elif self.provider == "openai":
                self.api_key = os.getenv("OPENAI_API_KEY")


@dataclass
class RetrievalConfig:
    """Configuration for retrieval"""
    top_k: int = 5
    similarity_threshold: float = 0.0
    distance_metric: str = "L2"  # L2, cosine, etc.


@dataclass
class MemoryConfig:
    """Configuration for conversational memory"""
    history_window_size: int = 5  # Number of turns to keep
    enabled: bool = True


@dataclass
class DataConfig:
    """Configuration for data paths"""
    data_dir: str = "data/pdfs"
    supported_extensions: list = field(default_factory=lambda: [".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"])


@dataclass
class Settings:
    """Main application settings"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def __post_init__(self):
        """Validate and set derived values"""
        # Ensure directories exist
        Path(self.data.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.vector_store.persist_dir).mkdir(parents=True, exist_ok=True)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get or create global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings():
    """Reset global settings (useful for testing)"""
    global _settings
    _settings = None



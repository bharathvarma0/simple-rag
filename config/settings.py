"""
Application settings and configuration
Simplified for adaptive RAG system
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
class VectorStoreConfig:
    """Configuration for vector store"""
    persist_dir: str = "vector_store"
    index_type: str = "hnsw"  # Options: "flat", "hnsw"
    hnsw_m: int = 32  # Number of connections per node for HNSW


@dataclass
class DataConfig:
    """Configuration for data paths"""
    data_dir: str = "data/pdfs"
    supported_extensions: list = field(default_factory=lambda: [".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"])


@dataclass
class Settings:
    """Main application settings"""
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
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


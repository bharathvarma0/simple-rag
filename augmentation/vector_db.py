"""
Vector database management using Qdrant
"""

from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
import uuid
from typing import List, Any, Dict, Optional, Tuple
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from components.embedders import EmbeddingGenerator
from components.chunkers import DocumentChunker
from utils.logger import get_logger
from utils.paths import ensure_dir
from config import get_settings

logger = get_logger(__name__)


class VectorDatabase:
    """Manage vector database using Qdrant"""
    
    def __init__(self, persist_dir: Optional[str] = None, embedding_model: Optional[str] = None):
        """
        Initialize vector database
        
        Args:
            persist_dir: Directory to persist the vector store (uses config default if None)
            embedding_model: Embedding model name (uses config default if None)
        """
        settings = get_settings()
        persist_dir = persist_dir or settings.vector_store.persist_dir
        embedding_model = embedding_model or settings.embedding.model_name
        
        self.persist_dir = ensure_dir(persist_dir)
        self.collection_name = "rag_collection"
        
        # Initialize Qdrant client
        if settings.vector_store.url:
            logger.info(f"Connecting to Qdrant at {settings.vector_store.url}")
            self.client = QdrantClient(
                url=settings.vector_store.url, 
                api_key=settings.vector_store.api_key
            )
        else:
            logger.info(f"Using local Qdrant at {self.persist_dir}")
            self.client = QdrantClient(path=str(self.persist_dir))
        
        self.embedding_model = embedding_model
        self.embedder = EmbeddingGenerator(model_name=embedding_model)
        
        logger.info(f"Vector database initialized at: {self.persist_dir}")
    
    def build_from_documents(
        self, 
        documents: List[Any], 
        chunk_size: Optional[int] = None, 
        chunk_overlap: Optional[int] = None
    ):
        """
        Build vector database from documents
        
        Args:
            documents: List of LangChain Document objects
            chunk_size: Size of text chunks (uses config default if None)
            chunk_overlap: Overlap between chunks (uses config default if None)
        """
        settings = get_settings()
        chunk_size = chunk_size or settings.chunking.chunk_size
        chunk_overlap = chunk_overlap or settings.chunking.chunk_overlap
        
        logger.info(f"Building vector database from {len(documents)} documents...")
        
        # Chunk documents
        chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_documents(documents)
        
        self.add_chunks(chunks)
        
        logger.info("Vector database built and saved")
        
    def add_chunks(self, chunks: List[Any]):
        """
        Add pre-chunked documents to the vector database
        
        Args:
            chunks: List of Document objects (chunks)
        """
        if not chunks:
            return
            
        logger.info(f"Adding {len(chunks)} chunks to vector database...")
        
        # Generate embeddings
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embedder.generate_embeddings(texts)
        
        # Prepare metadata
        metadatas = []
        for i, chunk in enumerate(chunks):
            meta = {
                "text": chunk.page_content,
                "chunk_id": i,
                **chunk.metadata
            }
            metadatas.append(meta)
        
        # Add to index
        self.add_embeddings(embeddings.astype('float32'), metadatas)
    
    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Dict] = None):
        """
        Add embeddings to the vector database
        
        Args:
            embeddings: NumPy array of embeddings
            metadatas: List of metadata dictionaries
        """
        dim = embeddings.shape[1]
        
        # Check if collection exists and has correct config
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if exists:
            # Check if dimension matches
            coll_info = self.client.get_collection(self.collection_name)
            if coll_info.config.params.vectors.size != dim:
                logger.warning(f"Collection dimension mismatch (expected {dim}, got {coll_info.config.params.vectors.size}). Recreating...")
                self.client.delete_collection(self.collection_name)
                exists = False
        
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )
            logger.info(f"Created collection '{self.collection_name}' with dim={dim}")
        
        # Prepare points
        points = []
        for i, vector in enumerate(embeddings):
            # Generate a UUID for the point if not provided
            point_id = str(uuid.uuid4())
            payload = metadatas[i] if metadatas else {}
            
            points.append(models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload=payload
            ))
            
        # Upsert in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )
        
        logger.info(f"Added {len(points)} vectors to database")
    
    def search(self, query_embedding: np.ndarray, k: int = 5, filter_ids: Optional[List[int]] = None) -> Tuple[List[float], List[int], List[Dict]]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector
            k: Number of results
            filter_ids: Optional list of chunk IDs to filter by (for optimization)
            
        Returns:
            Tuple of (distances, indices, metadatas)
            Note: indices here will be the chunk_ids from payload, not Qdrant UUIDs
        """
        search_filter = None
        if filter_ids is not None:
            # Create a filter for chunk_ids
            search_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="chunk_id",
                        match=models.MatchAny(any=filter_ids)
                    )
                ]
            )
            
        # Use query_points instead of search (which is deprecated/removed)
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding[0].tolist(),
            query_filter=search_filter,
            limit=k
        )
        
        results = response.points
        
        distances = []
        indices = []
        metadatas = []
        
        for res in results:
            distances.append(1 - res.score) # Convert cosine similarity to distance-like metric (smaller is better)
            # We use chunk_id from payload as the index for compatibility with existing code
            chunk_id = res.payload.get("chunk_id", -1)
            indices.append(chunk_id)
            metadatas.append(res.payload)
            
        return distances, indices, metadatas

    def save(self):
        """Save vector database to disk - Qdrant handles this automatically"""
        pass
    
    def load(self):
        """Load vector database from disk - Qdrant handles this automatically"""
        # Just verify collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        if exists:
            count = self.client.count(self.collection_name).count
            logger.info(f"Loaded vector database. Total vectors: {count}")
            # We need to populate self.metadata for compatibility with search.py
            # This is a bit expensive but needed for the current architecture
            # In a real prod system, we'd avoid loading all metadata into memory
            # For now, we'll load it lazily or just rely on search returning it
            self.metadata = [] # Placeholder, search() returns metadata directly now
        else:
            logger.warning(f"Collection '{self.collection_name}' not found")
    
    def exists(self) -> bool:
        """Check if vector database exists"""
        collections = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in collections)


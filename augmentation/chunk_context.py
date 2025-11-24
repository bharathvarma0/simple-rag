"""
Chunk Context Manager

Handles chunk relationships and retrieval of neighboring chunks
for multi-stage retrieval.
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ChunkContext:
    """Manages chunk relationships and context expansion"""
    
    def __init__(self, metadata_path: Optional[Path] = None):
        """
        Initialize chunk context manager
        
        Args:
            metadata_path: Path to store chunk relationship metadata
        """
        self.metadata_path = metadata_path or Path("vector_store/chunk_metadata.json")
        self.chunk_relationships = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load chunk relationship metadata"""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self.chunk_relationships = json.load(f)
                logger.info(f"Loaded chunk metadata: {len(self.chunk_relationships)} chunks")
            except Exception as e:
                logger.warning(f"Could not load chunk metadata: {e}")
                self.chunk_relationships = {}
    
    def _save_metadata(self):
        """Save chunk relationship metadata"""
        try:
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.chunk_relationships, f, indent=2)
            logger.info(f"Saved chunk metadata: {len(self.chunk_relationships)} chunks")
        except Exception as e:
            logger.error(f"Could not save chunk metadata: {e}")
    
    def index_chunks(self, chunks: List[Dict[str, Any]]):
        """
        Index chunks and build relationship map
        
        Args:
            chunks: List of chunks with metadata
        """
        # Group chunks by source document
        doc_chunks = {}
        for idx, chunk in enumerate(chunks):
            source = chunk.get('metadata', {}).get('source', 'unknown')
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append({
                'index': idx,
                'chunk_id': str(idx),
                'content': chunk.get('page_content') or chunk.get('content', ''),
                'metadata': chunk.get('metadata', {})
            })
        
        # Build relationships
        for source, source_chunks in doc_chunks.items():
            # Sort by position if available, otherwise by index
            source_chunks.sort(key=lambda x: x.get('metadata', {}).get('position', x['index']))
            
            for i, chunk in enumerate(source_chunks):
                chunk_id = chunk['chunk_id']
                self.chunk_relationships[chunk_id] = {
                    'source': source,
                    'position': i,
                    'total_chunks': len(source_chunks),
                    'prev_id': source_chunks[i-1]['chunk_id'] if i > 0 else None,
                    'next_id': source_chunks[i+1]['chunk_id'] if i < len(source_chunks)-1 else None,
                    'index': chunk['index']
                }
        
        self._save_metadata()
        logger.info(f"Indexed {len(self.chunk_relationships)} chunks from {len(doc_chunks)} documents")
    
    def get_neighboring_chunks(
        self,
        chunk: Dict[str, Any],
        chunks_collection: List[Dict[str, Any]],
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring chunks (before/after) from same document
        
        Args:
            chunk: The original chunk
            chunks_collection: Full collection of chunks
            depth: How many neighbors to retrieve (1 = prev+next, 2 = 2 prev + 2 next)
            
        Returns:
            List of neighboring chunks including the original
        """
        if depth < 1:
            return [chunk]
        
        # Get chunk ID (try different fields)
        chunk_text = chunk.get('page_content') or chunk.get('content', '')
        chunk_metadata = chunk.get('metadata', {})
        
        # Find chunk in relationships by content match
        chunk_id = None
        for cid, rel in self.chunk_relationships.items():
            if rel['index'] < len(chunks_collection):
                coll_chunk = chunks_collection[rel['index']]
                coll_text = coll_chunk.get('page_content') or coll_chunk.get('content', '')
                if coll_text == chunk_text:
                    chunk_id = cid
                    break
        
        if not chunk_id or chunk_id not in self.chunk_relationships:
            logger.debug("Chunk not found in relationships, returning original only")
            return [chunk]
        
        rel = self.chunk_relationships[chunk_id]
        neighbors = [chunk]  # Start with original
        
        # Get previous chunks
        current_id = rel['prev_id']
        for _ in range(depth):
            if current_id and current_id in self.chunk_relationships:
                prev_rel = self.chunk_relationships[current_id]
                if prev_rel['index'] < len(chunks_collection):
                    neighbors.insert(0, chunks_collection[prev_rel['index']])
                current_id = prev_rel['prev_id']
            else:
                break
        
        # Get next chunks
        current_id = rel['next_id']
        for _ in range(depth):
            if current_id and current_id in self.chunk_relationships:
                next_rel = self.chunk_relationships[current_id]
                if next_rel['index'] < len(chunks_collection):
                    neighbors.append(chunks_collection[next_rel['index']])
                current_id = next_rel['next_id']
            else:
                break
        
        logger.debug(f"Expanded 1 chunk to {len(neighbors)} chunks (depth={depth})")
        return neighbors
    
    def expand_chunks(
        self,
        chunks: List[Dict[str, Any]],
        chunks_collection: List[Dict[str, Any]],
        depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Expand all chunks with their neighbors
        
        Args:
            chunks: Chunks to expand
            chunks_collection: Full collection for neighbor lookup
            depth: Neighbor depth
            
        Returns:
            Expanded list of chunks (with duplicates removed)
        """
        if depth < 1:
            return chunks
        
        expanded = []
        seen_indices = set()
        
        for chunk in chunks:
            neighbors = self.get_neighboring_chunks(chunk, chunks_collection, depth)
            for neighbor in neighbors:
                # Use content hash to avoid duplicates
                content = neighbor.get('page_content') or neighbor.get('content', '')
                content_hash = hash(content)
                if content_hash not in seen_indices:
                    seen_indices.add(content_hash)
                    expanded.append(neighbor)
        
        logger.info(f"Expanded {len(chunks)} chunks to {len(expanded)} chunks (depth={depth})")
        return expanded

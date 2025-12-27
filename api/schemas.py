from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None

class Source(BaseModel):
    rank: int
    similarity_score: float
    distance: float
    source: str
    preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    context: str
    num_sources: int
    rewritten_query: Optional[str] = None
    strategy: Optional[str] = None
    complexity: Optional[str] = None

class IngestRequest(BaseModel):
    data_dir: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class IngestResponse(BaseModel):
    message: str
    num_documents: int
    num_chunks: int

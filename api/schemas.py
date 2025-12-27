from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    document_name: Optional[str] = None

class Source(BaseModel):
    rank: int
    distance: float
    source: str
    preview: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    context: str
    num_sources: int
    rewritten_query: Optional[str] = None

class IngestRequest(BaseModel):
    data_dir: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None

class IngestResponse(BaseModel):
    message: str
    num_documents: int
    num_chunks: int

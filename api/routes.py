from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from adaptive_rag import AdaptiveRAGPipeline
from data_ingestion.ingest import ingest_documents
from augmentation.vector_db import VectorDatabase
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


# Global instance
rag_pipeline = None

def get_rag_pipeline():
    global rag_pipeline
    if rag_pipeline is None:
        rag_pipeline = AdaptiveRAGPipeline()
    return rag_pipeline

import uuid

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the RAG system
    """
    try:
        rag = get_rag_pipeline()
        result = rag.query(request.question, doc_id=request.doc_id)
        
        # Add complexity from query profile if available
        if 'query_profile' in result:
            result['complexity'] = result['query_profile'].get('complexity')
            
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@router.get("/documents")
def list_documents():
    """
    List available documents
    """
    try:
        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        
        if not data_dir.exists():
            return {"documents": []}
            
        # Get all supported files
        files = []
        # glob is case-sensitive on Linux/Mac usually, but let's be safe
        # Also, some extensions might be uppercase in the config or filesystem
        for ext in settings.data.supported_extensions:
            # Try both lowercase and uppercase extension
            files.extend(list(data_dir.glob(f"**/*{ext}")))
            files.extend(list(data_dir.glob(f"**/*{ext.upper()}")))
            
        # Remove duplicates if any (e.g. if ext was already uppercase)
        files = list(set(files))
            
        return {
            "documents": [f.name for f in files],
            "count": len(files)
        }
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import UploadFile, File
import shutil

@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """
    Upload a document to the data directory
    """
    try:
        settings = get_settings()
        data_dir = Path(settings.data.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = data_dir / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Generate a unique Doc ID
        doc_id = str(uuid.uuid4())
            
        # Trigger automatic ingestion for this file
        try:
            # 1. Load and chunk just this file
            from components.loaders import PDFLoader
            from components.chunkers import DocumentChunker
            
            # Load
            loader = PDFLoader(file_path)
            documents = loader.load()
            
            if documents:
                # Add metadata (doc_id and filename) to every document/page BEFORE chunking
                for doc in documents:
                    doc.metadata["doc_id"] = doc_id
                    doc.metadata["filename"] = file.filename
                
                # Chunk
                chunker = DocumentChunker(
                    chunk_size=settings.chunking.chunk_size,
                    chunk_overlap=settings.chunking.chunk_overlap
                )
                chunks = chunker.chunk_documents(documents)
                
                # Add to Vector DB
                # Use the global RAG pipeline's vector DB to avoid lock contention
                rag = get_rag_pipeline()
                if hasattr(rag, 'vector_db'):
                    rag.vector_db.add_chunks(chunks)
                else:
                    logger.error("Could not access vector_db from global pipeline")
                    raise RuntimeError("Global vector DB access failed")
                
                logger.info(f"Automatically ingested {len(chunks)} chunks for {file.filename} (doc_id={doc_id})")
                
        except Exception as e:
            logger.error(f"Automatic ingestion failed for {file.filename}: {e}")
            return {"message": f"Uploaded {file.filename}, but ingestion failed: {str(e)}", "filename": file.filename}
            
        return {
            "message": f"Successfully uploaded and ingested {file.filename}", 
            "filename": file.filename,
            "doc_id": doc_id
        }
    except Exception as e:
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

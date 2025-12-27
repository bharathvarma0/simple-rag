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

@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Query the RAG system
    """
    try:
        rag = get_rag_pipeline()
        result = rag.query(request.question)
        
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
            
        # Trigger automatic ingestion for this file
        try:
            # 1. Load and chunk just this file
            # We need to modify ingest_documents to accept a specific file, 
            # or we can use the lower-level components directly here for speed.
            # Let's use the components directly to avoid re-scanning the whole dir.
            
            from components.loaders import PDFLoader
            from components.chunkers import DocumentChunker
            from augmentation.vector_db import VectorDatabase
            from augmentation.keyword_db import KeywordDatabase
            
            # Load
            loader = PDFLoader(file_path)
            documents = loader.load()
            
            if documents:
                # Chunk
                chunker = DocumentChunker(
                    chunk_size=settings.chunking.chunk_size,
                    chunk_overlap=settings.chunking.chunk_overlap
                )
                chunks = chunker.chunk_documents(documents)
                
                # Add to Vector DB
                # Use the global RAG pipeline's vector DB to avoid lock contention
                rag = get_rag_pipeline()
                # We need to access the vector_db from the pipeline. 
                # AdaptiveRAGPipeline -> vector_db attribute
                if hasattr(rag, 'vector_db'):
                    rag.vector_db.add_chunks(chunks)
                else:
                    # Fallback if structure is different (shouldn't happen based on code)
                    logger.error("Could not access vector_db from global pipeline")
                    raise RuntimeError("Global vector DB access failed")
                
                # Update Keyword DB (Note: This is still not ideal for concurrency, but works for now)
                # Ideally we'd append to the index, but BM25 usually needs a full rebuild or complex incremental updates.
                # For now, we'll skip rebuilding the WHOLE BM25 index on every upload to avoid blocking.
                # A background task should handle full re-indexing.
                
                logger.info(f"Automatically ingested {len(chunks)} chunks for {file.filename}")
                
        except Exception as e:
            logger.error(f"Automatic ingestion failed for {file.filename}: {e}")
            # We don't fail the upload if ingestion fails, but we warn
            return {"message": f"Uploaded {file.filename}, but ingestion failed: {str(e)}", "filename": file.filename}
            
        return {"message": f"Successfully uploaded and ingested {file.filename}", "filename": file.filename}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from api.schemas import QueryRequest, QueryResponse, IngestRequest, IngestResponse
from generation.rag import RAGPipeline
from data_ingestion.ingest import ingest_documents
from augmentation.vector_db import VectorDatabase
from config import get_settings
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system
    """
    try:
        rag = RAGPipeline()
        result = rag.query(
            request.question, 
            top_k=request.top_k,
            document_name=request.document_name
        )
        return result
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Trigger document ingestion and vector store build
    """
    try:
        settings = get_settings()
        data_dir = request.data_dir or settings.data.data_dir
        chunk_size = request.chunk_size or settings.chunking.chunk_size
        chunk_overlap = request.chunk_overlap or settings.chunking.chunk_overlap
        
        # Ingest documents
        try:
            raw_docs, chunks = ingest_documents(
                data_dir=data_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
            
        # Build vector database
        vector_db = VectorDatabase(
            persist_dir=settings.vector_store.persist_dir,
            embedding_model=settings.embedding.model_name
        )
        vector_db.build_from_documents(
            raw_docs,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        return {
            "message": "Ingestion and vector store build completed successfully",
            "num_documents": len(raw_docs) if raw_docs else 0,
            "num_chunks": len(chunks) if chunks else 0
        }
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

@router.get("/documents")
async def list_documents():
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
        for ext in settings.data.supported_extensions:
            files.extend(list(data_dir.glob(f"**/*{ext}")))
            
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
async def upload_document(file: UploadFile = File(...)):
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
            
        # Auto-ingest the uploaded file
        try:
            logger.info(f"Auto-ingesting uploaded file: {file.filename}")
            raw_docs, chunks = ingest_documents(
                data_dir=str(data_dir),
                only_files=[file.filename]
            )
            
            if chunks:
                vector_db = VectorDatabase(
                    persist_dir=settings.vector_store.persist_dir,
                    embedding_model=settings.embedding.model_name
                )
                vector_db.build_from_documents(raw_docs) # Note: This appends if collection exists
                logger.info(f"Successfully ingested {file.filename}")
                
        except Exception as e:
            logger.error(f"Auto-ingestion failed for {file.filename}: {e}")
            # We don't fail the upload if ingestion fails, but we log it
            
        return {"message": f"Successfully uploaded and ingested {file.filename}", "filename": file.filename}
    except Exception as e:
        logger.error(f"Failed to upload document: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

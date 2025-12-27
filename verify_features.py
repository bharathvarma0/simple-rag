import sys
from pathlib import Path
import shutil
import time

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from generation.rag import RAGPipeline
from data_ingestion.ingest import ingest_documents
from utils.logger import setup_logging
from config import get_settings

setup_logging()

def test_features():
    print("Initializing Test...")
    settings = get_settings()
    data_dir = Path(settings.data.data_dir)
    
    # 1. Simulate Upload and Auto-Ingest
    print("\n[Test 1] Auto-Ingest")
    test_file = data_dir / "test_doc.pdf"
    
    # Create a dummy PDF if not exists (or copy one)
    # For this test, we'll assume there's at least one PDF in data_dir we can copy
    existing_pdfs = [p for p in data_dir.glob("*.pdf") if p.name != "test_doc.pdf"]
    if not existing_pdfs:
        print("No PDFs found to test copy. Skipping Auto-Ingest test.")
    else:
        source_pdf = existing_pdfs[0]
        if test_file.exists():
            test_file.unlink()
        shutil.copy(source_pdf, test_file)
        print(f"Created test file: {test_file.name}")
        
        # Simulate auto-ingest
        print(f"Triggering ingestion for only: {test_file.name}")
        raw_docs, chunks = ingest_documents(data_dir=str(data_dir), only_files=[test_file.name])
        
        if chunks:
            from augmentation.vector_db import VectorDatabase
            vector_db = VectorDatabase(
                persist_dir=settings.vector_store.persist_dir,
                embedding_model=settings.embedding.model_name
            )
            vector_db.build_from_documents(raw_docs)
            print("Auto-ingest complete.")
            # Explicitly delete to release lock
            del vector_db

    # 2. Test Query Filtering
    print("\n[Test 2] Query Filtering")
    # Initialize RAG Pipeline after releasing lock
    rag = RAGPipeline()
    
    query = "regulations" # Generic query
    
    # Query without filter
    print("Querying WITHOUT filter...")
    res_all = rag.query(query, top_k=3)
    sources_all = [s["source"] for s in res_all["sources"]]
    print(f"Sources found: {sources_all}")
    
    # Query WITH filter (using the test file if created, else the first existing one)
    target_doc = test_file.name if test_file.exists() else existing_pdfs[0].name
    print(f"Querying WITH filter: document_name='{target_doc}'")
    
    res_filter = rag.query(query, top_k=3, document_name=target_doc)
    sources_filter = [s["source"] for s in res_filter["sources"]]
    print(f"Sources found: {sources_filter}")
    
    # Verification
    if all(target_doc in s for s in sources_filter):
        print("SUCCESS: All filtered results match target document.")
    else:
        print(f"FAILED: Filtered results contain other documents: {sources_filter}")
        sys.exit(1)

    # Cleanup
    if test_file.exists():
        test_file.unlink()
        print(f"Cleaned up test file: {test_file.name}")

if __name__ == "__main__":
    test_features()

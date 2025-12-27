"""
Complete End-to-End RAG Pipeline
Runs: Ingestion → Chunking → Embedding → Vector DB → Evaluation

This script demonstrates the full pipeline from raw documents to evaluation results.
"""

import sys
from pathlib import Path
import shutil
import time

sys.path.append(str(Path(__file__).parent))

from data_ingestion.ingest import ingest_documents
from augmentation.vector_db import VectorDatabase
from evaluation.evaluate_adaptive import run_evaluation
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def print_header(title):
    """Print a formatted header"""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def print_step(step_num, title):
    """Print a step header"""
    print()
    print(f"{'='*80}")
    print(f"STEP {step_num}: {title}")
    print(f"{'='*80}")
    print()


def clean_vector_store():
    """Clean old vector database"""
    print_step(1, "Cleaning Old Vector Database")
    
    vector_store_path = Path("vector_store")
    
    if vector_store_path.exists():
        # Remove all files but keep directory
        for item in vector_store_path.iterdir():
            if item.is_file():
                item.unlink()
                print(f"  ✅ Deleted: {item.name}")
        print()
        print("✅ Old vector database cleaned successfully")
    else:
        vector_store_path.mkdir(exist_ok=True)
        print("✅ Created vector_store directory")
    
    print()


def ingest_and_chunk():
    """Ingest documents and create chunks"""
    print_step(2, "Ingesting Documents & Creating Chunks")
    
    print("📄 Reading PDFs from data/pdfs/...")
    print()
    
    start_time = time.time()
    docs, chunks = ingest_documents()
    elapsed = time.time() - start_time
    
    print()
    print(f"✅ Ingestion Complete!")
    print(f"   Documents: {len(docs)}")
    print(f"   Chunks: {len(chunks)}")
    print(f"   Time: {elapsed:.2f}s")
    print()
    
    # Show sample chunk info
    if chunks:
        print("📊 Sample Chunk Info:")
        sample = chunks[0]
        print(f"   Source: {sample.metadata.get('source', 'unknown')}")
        print(f"   Length: {len(sample.page_content)} characters")
        print(f"   Preview: {sample.page_content[:100]}...")
    
    print()
    return docs, chunks


def build_vector_database(docs):
    """Build and save vector database"""
    print_step(3, "Building Vector Database (Embedding)")
    
    print("🔢 Generating embeddings and building FAISS index...")
    print("   This may take a few minutes depending on document count...")
    print()
    
    start_time = time.time()
    vdb = VectorDatabase()
    vdb.build_from_documents(docs)
    elapsed = time.time() - start_time
    
    print()
    print(f"✅ Vector Database Built Successfully!")
    print(f"   Time: {elapsed:.2f}s")
    print(f"   Location: vector_store/")
    print()
    
    return vdb


def run_evaluation_pipeline():
    """Run evaluation on the built system"""
    print_step(4, "Running Evaluation")
    
    print("🧪 Testing adaptive RAG system with evaluation questions...")
    print("📝 Results will be automatically saved to evaluation/outputs/")
    print()
    
    # Run evaluation (displays on screen AND saves to file)
    run_evaluation()
    
    print()
    print("✅ Evaluation complete! Results saved to evaluation/outputs/")
    print()


def main():
    """Run complete pipeline"""
    
    print_header("COMPLETE RAG PIPELINE - END TO END")
    print("This script will:")
    print("  1. Clean old vector database")
    print("  2. Ingest documents from data/pdfs/")
    print("  3. Chunk documents into smaller pieces")
    print("  4. Generate embeddings and build vector database")
    print("  5. Run evaluation with 20 test questions")
    print()
    print("Expected time: 3-5 minutes")
    print()
    
    input("Press Enter to start the pipeline...")
    
    overall_start = time.time()
    
    try:
        # Step 1: Clean
        clean_vector_store()
        
        # Step 2: Ingest & Chunk
        docs, chunks = ingest_and_chunk()
        
        if not docs:
            print("❌ ERROR: No documents found in data/pdfs/")
            print("   Please add PDF files to data/pdfs/ directory")
            return
        
        # Step 3: Build Vector DB
        vdb = build_vector_database(docs)
        
        # Step 4: Evaluate
        run_evaluation_pipeline()
        
        # Final summary
        overall_elapsed = time.time() - overall_start
        
        print()
        print("=" * 80)
        print("🎉 PIPELINE COMPLETE!")
        print("=" * 80)
        print()
        print(f"Total Time: {overall_elapsed:.2f}s ({overall_elapsed/60:.1f} minutes)")
        print()
        print("✅ Vector database built and saved")
        print("✅ Evaluation results displayed above")
        print()
        print("Next steps:")
        print("  - Review classification accuracy in the summary")
        print("  - Check which strategies were selected")
        print("  - Tune config/strategies.yaml if needed")
        print()
        print("=" * 80)
        
    except KeyboardInterrupt:
        print()
        print("⚠️  Pipeline interrupted by user")
        print()
    except Exception as e:
        print()
        print(f"❌ ERROR: Pipeline failed: {e}")
        print()
        logger.exception("Pipeline failed")


if __name__ == "__main__":
    main()

import sys
from pathlib import Path
import json

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from generation.rag import RAGPipeline
from api.schemas import QueryResponse
from utils.logger import setup_logging

setup_logging()

def test_query_schema():
    print("Initializing RAG Pipeline...")
    try:
        rag = RAGPipeline()
        print("RAG Pipeline initialized.")
        
        query = "What are the financial regulations?"
        print(f"Querying: {query}")
        
        result = rag.query(query, top_k=1)
        print("Query successful!")
        
        # Validate against Pydantic model
        try:
            response = QueryResponse(**result)
            print("Response validation passed!")
            print(json.dumps(result, indent=2, default=str)[:500] + "...")
        except Exception as e:
            print(f"Response validation FAILED: {e}")
            sys.exit(1)
            
    except Exception as e:
        print(f"FAILED: {e}")
        # import traceback
        # traceback.print_exc()

if __name__ == "__main__":
    test_query_schema()

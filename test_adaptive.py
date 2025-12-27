"""
Simple test script for adaptive RAG system
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from adaptive_rag import AdaptiveRAGPipeline
from utils.logger import setup_logging

setup_logging()


def main():
    """Test adaptive RAG with a few sample questions"""
    
    print("=" * 60)
    print("Adaptive RAG System - Quick Test")
    print("=" * 60)
    print()
    
    # Initialize
    try:
        rag = AdaptiveRAGPipeline()
        print("✓ Adaptive RAG initialized successfully")
        print()
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        return
    
    # Test questions
    test_questions = [
        "What is the maximum fuel flow rate allowed in the 2026 regulations?",
        "Summarize the key changes in the power unit regulations for 2026.",
        "Compare the financial restrictions on power unit manufacturers vs. chassis constructors."
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"Question {i}: {question}")
        print("-" * 60)
        
        try:
            result = rag.query(question)
            
            print(f"Strategy: {result.get('strategy')}")
            print(f"Query Type: {result.get('query_profile', {}).get('query_type')}")
            print(f"Complexity: {result.get('query_profile', {}).get('complexity')}")
            print(f"Response Time: {result.get('response_time', 0):.2f}s")
            print()
            print("Answer:")
            print(result['answer'])
            print()
            print("=" * 60)
            print()
            
        except Exception as e:
            print(f"Error: {e}")
            print()
    
    # Show metrics
    metrics = rag.get_metrics()
    print("Session Metrics:")
    print(f"  Total Queries: {metrics.get('total_queries', 0)}")
    print(f"  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
    print(f"  Strategy Distribution: {metrics.get('strategy_distribution', {})}")
    print()


if __name__ == "__main__":
    main()

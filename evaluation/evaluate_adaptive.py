"""
Evaluation script for adaptive RAG system
Compatible with existing evaluation questions
"""

import sys
from pathlib import Path
import time
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))

from adaptive_rag import AdaptiveRAGPipeline
from evaluation.questions import EVALUATION_QUESTIONS
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


def run_evaluation():
    """Run evaluation on adaptive RAG system"""
    
    print("=" * 80)
    print("Adaptive RAG System Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Initialize adaptive RAG
    try:
        rag = AdaptiveRAGPipeline()
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG system: {e}")
        return
    
    # Results storage
    results = []
    total_time = 0
    
    # Run evaluation
    for i, item in enumerate(EVALUATION_QUESTIONS, 1):
        question = item['question']
        category = item['category']
        
        print(f"Question {i}: {question}")
        print(f"Category: {category}")
        
        start_time = time.time()
        
        try:
            result = rag.query(question)
            time_taken = time.time() - start_time
            total_time += time_taken
            
            print(f"Time Taken: {time_taken:.2f}s")
            print("-" * 40)
            print("Answer:")
            print(result['answer'])
            print("-" * 40)
            print("Metadata:")
            print(f"  Strategy: {result.get('strategy', 'N/A')}")
            print(f"  Query Type: {result.get('query_profile', {}).get('query_type', 'N/A')}")
            print(f"  Complexity: {result.get('query_profile', {}).get('complexity', 'N/A')}")
            print(f"  Sources: {result.get('num_sources', 0)}")
            print("-" * 40)
            print("Sources:")
            for source in result.get('sources', [])[:3]:  # Show top 3
                print(f"  - {source['source']} (Score: {source['similarity_score']:.3f})")
            print()
            print("=" * 80)
            print()
            
            results.append({
                'question': question,
                'category': category,
                'answer': result['answer'],
                'strategy': result.get('strategy'),
                'query_type': result.get('query_profile', {}).get('query_type'),
                'time_taken': time_taken,
                'num_sources': result.get('num_sources', 0)
            })
            
        except Exception as e:
            print(f"[ERROR] Failed to process question: {e}")
            print()
            print("=" * 80)
            print()
    
    # Print summary
    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print(f"Average Time per Question: {total_time / len(EVALUATION_QUESTIONS):.2f}s")
    print(f"Total Time: {total_time:.2f}s")
    print("=" * 80)
    print()
    
    # Print strategy distribution
    print("Strategy Distribution:")
    strategy_counts = {}
    for r in results:
        strategy = r.get('strategy', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {count} ({count/len(results)*100:.1f}%)")
    print()
    
    # Print query type distribution
    print("Query Type Distribution:")
    type_counts = {}
    for r in results:
        qtype = r.get('query_type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {qtype}: {count} ({count/len(results)*100:.1f}%)")
    print()
    
    # Get system metrics
    metrics = rag.get_metrics()
    print("System Metrics:")
    print(f"  Total Queries: {metrics.get('total_queries', 0)}")
    print(f"  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
    print()


if __name__ == "__main__":
    run_evaluation()

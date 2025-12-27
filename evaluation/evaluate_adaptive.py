"""
Evaluation script for adaptive RAG system
Compatible with existing evaluation questions
"""

import sys
from pathlib import Path
import time
from datetime import datetime
import io

# Fix Unicode encoding for Windows when redirecting to file
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.append(str(Path(__file__).parent.parent))

from adaptive_rag import AdaptiveRAGPipeline
from evaluation.questions import EVALUATION_QUESTIONS
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

# Mapping from manual categories to expected query types
CATEGORY_TO_TYPE = {
    "Fact Retrieval": "fact",
    "Summarization": "summary",
    "Complex Reasoning": "reasoning",
    "Context Understanding": "fact",  # Could be fact or reasoning
}


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
    classification_correct = 0
    classification_total = 0
    
    # Run evaluation
    for i, item in enumerate(EVALUATION_QUESTIONS, 1):
        question = item['question']
        category = item['category']
        
        print(f"Question {i}: {question}")
        print(f"Expected Category: {category}")
        
        start_time = time.time()
        
        try:
            result = rag.query(question)
            time_taken = time.time() - start_time
            total_time += time_taken
            
            # Check classification accuracy
            identified_type = result.get('query_profile', {}).get('query_type', 'unknown')
            expected_type = CATEGORY_TO_TYPE.get(category, 'unknown')
            is_correct = (identified_type == expected_type)
            
            if is_correct:
                classification_correct += 1
            classification_total += 1
            
            # Show classification comparison
            status = "✅" if is_correct else "❌"
            print(f"{status} Identified Type: {identified_type} (Expected: {expected_type})")
            print(f"Time Taken: {time_taken:.2f}s")
            print("-" * 40)
            print("Answer:")
            print(result['answer'])
            print("-" * 40)
            print("Metadata:")
            print(f"  Strategy: {result.get('strategy', 'N/A')}")
            print(f"  Query Type: {identified_type}")
            print(f"  Complexity: {result.get('query_profile', {}).get('complexity', 'N/A')}")
            print(f"  Sources: {result.get('num_sources', 0)}")
            print("-" * 40)
            print("Top Sources:")
            for source in result.get('sources', [])[:3]:  # Show top 3
                print(f"  - {source['source']} (Score: {source['similarity_score']:.3f})")
            print()
            print("=" * 80)
            print()
            
            results.append({
                'question': question,
                'category': category,
                'expected_type': expected_type,
                'identified_type': identified_type,
                'classification_correct': is_correct,
                'answer': result['answer'],
                'strategy': result.get('strategy'),
                'complexity': result.get('query_profile', {}).get('complexity'),
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
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    
    # Classification accuracy
    classification_accuracy = (classification_correct / classification_total) * 100
    print(f"📊 Classification Accuracy: {classification_correct}/{classification_total} ({classification_accuracy:.1f}%)")
    print()
    
    # Breakdown by category
    print("Classification by Category:")
    category_stats = {}
    for r in results:
        cat = r['category']
        if cat not in category_stats:
            category_stats[cat] = {'correct': 0, 'total': 0}
        category_stats[cat]['total'] += 1
        if r['classification_correct']:
            category_stats[cat]['correct'] += 1
    
    for cat, stats in sorted(category_stats.items()):
        acc = (stats['correct'] / stats['total']) * 100
        status = "✅" if acc >= 80 else "⚠️" if acc >= 60 else "❌"
        print(f"  {status} {cat}: {stats['correct']}/{stats['total']} ({acc:.1f}%)")
    print()
    
    # Performance metrics
    print(f"⏱️  Average Time per Question: {total_time / len(EVALUATION_QUESTIONS):.2f}s")
    print(f"⏱️  Total Time: {total_time:.2f}s")
    print()
    
    # Strategy distribution
    print("Strategy Distribution:")
    strategy_counts = {}
    for r in results:
        strategy = r.get('strategy', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {strategy}: {count} ({count/len(results)*100:.1f}%)")
    print()
    
    # Query type distribution
    print("Query Type Distribution:")
    type_counts = {}
    for r in results:
        qtype = r.get('identified_type', 'unknown')
        type_counts[qtype] = type_counts.get(qtype, 0) + 1
    
    for qtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {qtype}: {count} ({count/len(results)*100:.1f}%)")
    print()
    
    # System metrics
    metrics = rag.get_metrics()
    print("System Metrics:")
    print(f"  Total Queries: {metrics.get('total_queries', 0)}")
    print(f"  Avg Response Time: {metrics.get('avg_response_time', 0):.2f}s")
    print()
    
    print("=" * 80)
    print()
    
    # Save results to file
    save_results_to_file(results)
    
    return results


def save_results_to_file(results):
    """Save clean results to evaluation/outputs folder"""
    from pathlib import Path
    
    # Create outputs directory
    output_dir = Path("evaluation/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = output_dir / f"evaluation_results_{timestamp}.txt"
    
    print(f"💾 Saving results to {filepath}...")
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            # Header
            f.write("=" * 80 + "\n")
            f.write("ADAPTIVE RAG EVALUATION RESULTS\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write each result
            for i, r in enumerate(results, 1):
                f.write(f"Q{i}: {r['question']}\n")
                f.write(f"Category: {r['category']}\n")
                status = "✓" if r['classification_correct'] else "✗"
                f.write(f"Classification: {status} (Expected: {r['expected_type']}, Got: {r['identified_type']})\n")
                f.write(f"Strategy: {r['strategy']}\n")
                f.write(f"\nAnswer:\n{r['answer']}\n")
                f.write("\n" + "-" * 80 + "\n\n")
            
            # Summary
            correct = sum(1 for r in results if r['classification_correct'])
            total = len(results)
            accuracy = (correct / total) * 100
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Overall Accuracy: {correct}/{total} ({accuracy:.1f}%)\n\n")
            
            # By category
            f.write("By Category:\n")
            category_stats = {}
            for r in results:
                cat = r['category']
                if cat not in category_stats:
                    category_stats[cat] = {'correct': 0, 'total': 0}
                category_stats[cat]['total'] += 1
                if r['classification_correct']:
                    category_stats[cat]['correct'] += 1
            
            for cat, stats in sorted(category_stats.items()):
                acc = (stats['correct'] / stats['total']) * 100
                f.write(f"  {cat}: {stats['correct']}/{stats['total']} ({acc:.0f}%)\n")
            f.write("\n")
            
            # Strategy usage
            f.write("Strategy Usage:\n")
            strategy_counts = {}
            for r in results:
                strategy = r.get('strategy', 'unknown')
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
            for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
                f.write(f"  {strategy}: {count} ({count/len(results)*100:.0f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"✅ Results saved to {filepath}")
        print()
        
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")
        print()


if __name__ == "__main__":
    run_evaluation()

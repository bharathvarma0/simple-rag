"""
Evaluation script for RAG system
Runs a set of questions and saves results to a file.
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from generation.rag import RAGPipeline
from evaluation.questions import EVALUATION_QUESTIONS

def run_evaluation():
    print("Initializing RAG Pipeline...")
    rag = RAGPipeline()
    
    output_file = Path("evaluation/evaluation_results.txt")
    
    print(f"Starting evaluation of {len(EVALUATION_QUESTIONS)} questions...")
    print(f"Results will be saved to: {output_file}")
    
    with open(output_file, "w", encoding="utf-8") as f:
        # Write Header
        f.write(f"RAG System Evaluation\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        total_time = 0
        
        for i, item in enumerate(EVALUATION_QUESTIONS, 1):
            category = item["category"]
            question = item["question"]
            
            print(f"\n[{i}/{len(EVALUATION_QUESTIONS)}] Processing ({category}): {question}")
            
            # Measure time
            start_time = time.time()
            response = rag.query(question)
            end_time = time.time()
            
            duration = end_time - start_time
            total_time += duration
            
            # Write to file
            f.write(f"Question {i}: {question}\n")
            f.write(f"Category: {category}\n")
            f.write(f"Time Taken: {duration:.2f}s\n")
            f.write("-" * 20 + "\n")
            f.write(f"Answer:\n{response['answer']}\n")
            f.write("-" * 20 + "\n")
            f.write("Sources:\n")
            for source in response['sources']:
                f.write(f"- {source['source']} (Score: {source['similarity_score']:.3f})\n")
            f.write("\n" + "="*80 + "\n\n")
            
            # Flush to ensure data is written even if script stops
            f.flush()
            
        avg_time = total_time / len(EVALUATION_QUESTIONS)
        summary = f"\nEvaluation Complete!\nAverage Time per Question: {avg_time:.2f}s\nTotal Time: {total_time:.2f}s\n"
        print(summary)
        f.write(summary)

if __name__ == "__main__":
    run_evaluation()

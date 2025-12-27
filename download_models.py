
import os
from sentence_transformers import SentenceTransformer

# Download reranker model (used locally)
print("Downloading reranker model...")
reranker_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
try:
    SentenceTransformer(reranker_name)
    print("Reranker model downloaded.")
except Exception as e:
    print(f"Error downloading model: {e}")

#!/usr/bin/env python3
"""
Simple model download script using stable versions
"""

import os
from sentence_transformers import SentenceTransformer

def download_model():
    print("Downloading model...")
    
    # Create models directory
    models_dir = "/app/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Download the model - using a smaller, reliable model
    model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder=models_dir)
    
    print("Model downloaded successfully!")
    
    # Quick test
    test_embedding = model.encode("test")
    print(f"Model working! Embedding size: {len(test_embedding)}")

if __name__ == "__main__":
    download_model()
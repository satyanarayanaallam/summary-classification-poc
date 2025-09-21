#!/usr/bin/env python3
"""Test script to verify Sentence Transformers integration."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from db.vector_store import FaissVectorStore
from services.triplet_service import TripletService
from utils.normalization import triplet_to_text

def test_sentence_transformers():
    """Test the Sentence Transformers integration."""
    print("Testing Sentence Transformers integration...")
    
    # Test 1: Initialize vector store
    print("\n1. Initializing FaissVectorStore with Sentence Transformers...")
    try:
        vector_store = FaissVectorStore(model_name="all-MiniLM-L6-v2")
        print(f"✓ Vector store initialized successfully")
        print(f"✓ Model dimension: {vector_store._dim}")
    except Exception as e:
        print(f"✗ Failed to initialize vector store: {e}")
        return False
    
    # Test 2: Test triplet extraction and normalization
    print("\n2. Testing triplet extraction and normalization...")
    try:
        triplet_service = TripletService()
        test_summary = "Payment of $1200 was made by ACME Corp on 2025-09-01 for invoice #INV-100."
        triplets = triplet_service.extract_and_normalize(test_summary)
        print(f"✓ Extracted {len(triplets)} triplets:")
        for i, triplet in enumerate(triplets):
            print(f"  {i+1}. {triplet}")
    except Exception as e:
        print(f"✗ Failed to extract triplets: {e}")
        return False
    
    # Test 3: Test embedding generation
    print("\n3. Testing embedding generation...")
    try:
        test_texts = [triplet_to_text(t) for t in triplets]
        print(f"✓ Triplet texts: {test_texts}")
        
        # Add to vector store
        test_metas = [{"doc_type": "INVOICE", "doc_code": "INV001", "test": True} for _ in test_texts]
        vector_store.add(test_texts, test_metas)
        print(f"✓ Added {len(test_texts)} triplets to vector store")
    except Exception as e:
        print(f"✗ Failed to generate embeddings: {e}")
        return False
    
    # Test 4: Test similarity search
    print("\n4. Testing similarity search...")
    try:
        query_text = "invoice has_amount <AMOUNT>"
        results = vector_store.query(query_text, top_k=3)
        print(f"✓ Query: '{query_text}'")
        print(f"✓ Found {len(results)} results:")
        for i, result in enumerate(results):
            print(f"  {i+1}. Score: {result.get('score', 0):.4f}, Doc Type: {result.get('doc_type', 'N/A')}")
    except Exception as e:
        print(f"✗ Failed to perform similarity search: {e}")
        return False
    
    print("\n✓ All tests passed! Sentence Transformers integration is working correctly.")
    return True

if __name__ == "__main__":
    success = test_sentence_transformers()
    sys.exit(0 if success else 1)

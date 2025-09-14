# scripts/test_retriever.py
import sys
import os
import logging

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.retriever import retrieve_context
from src.config import CONTEXT_RETRIEVAL_THRESHOLD

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_retrieval():
    """Performs a direct retrieval test with a known query."""
    
    # This is a known question from the 'medical_faqs.csv' dataset
    query = "What is Aphasia?"
    
    logging.info(f"--- Testing retrieval with query: '{query}' ---")
    
    # We call the retriever with the threshold disabled to see the raw results
    retrieved_docs = retrieve_context(query, threshold=0.0)
    
    if not retrieved_docs:
        logging.error("TEST FAILED: The retriever returned an empty list.")
        logging.error("This indicates a fundamental problem with the vector search.")
    else:
        logging.info("TEST PASSED: The retriever successfully found documents.")
        print("\n--- Retrieved Documents ---")
        for i, doc in enumerate(retrieved_docs):
            print(f"\nResult {i+1}:")
            print(f"  Text: {doc['text'][:250]}...")
            print(f"  Metadata: {doc['metadata']}")
            # The distance is not returned by the current retrieve_context, but this is fine for now.

if __name__ == "__main__":
    test_retrieval()

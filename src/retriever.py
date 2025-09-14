import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from src.config import (
    DB_PATH, 
    COLLECTION_NAME, 
    EMBEDDING_MODEL_NAME, 
    CONTEXT_RETRIEVAL_N_RESULTS, 
    CONTEXT_RETRIEVAL_THRESHOLD
)

def retrieve_context(
    query: str, 
    collection_name: str = COLLECTION_NAME, 
    db_path: Optional[str] = DB_PATH,
    client: Optional[chromadb.Client] = None,
    model_name: str = EMBEDDING_MODEL_NAME,
    n_results: int = CONTEXT_RETRIEVAL_N_RESULTS,
    threshold: float = CONTEXT_RETRIEVAL_THRESHOLD
) -> List[str]:
    """
    Retrieves relevant context from a ChromaDB vector store.

    Args:
        query: The user's query.
        collection_name: The name of the collection to query.
        db_path: Path for the persistent database. If None, a client instance must be provided.
        client: An optional chromadb.Client instance.
        model_name: The Sentence Transformers model to use.
        n_results: The number of documents to retrieve.
        threshold: The maximum distance score for relevance.

    Returns:
        A list of relevant document texts.
    """
    if client is None:
        if db_path:
            logging.info(f"Initializing ChromaDB persistent client at path: {db_path}")
            client = chromadb.PersistentClient(path=db_path)
        else:
            raise ValueError("Either a 'db_path' or a 'client' instance must be provided.")
    else:
        logging.info("Using provided ChromaDB client.")

    try:
        collection = client.get_collection(name=collection_name)
        logging.info(f"Querying collection: '{collection_name}'")
    except Exception as e:
        logging.error(f"Failed to get collection '{collection_name}': {e}")
        return []

    model = SentenceTransformer(model_name)
    query_embedding = model.encode(query).tolist()

    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    
    documents = results.get('documents', [[]])[0]
    distances = results.get('distances', [[]])[0]

    if not documents:
        return []

    if threshold > 0.0:
        return [doc for doc, dist in zip(documents, distances) if dist <= threshold]
    
    return documents

if __name__ == '__main__':
    test_query_english = "What are the symptoms of the flu?"
    print(f"\n--- Querying with: '{test_query_english}' ---")
    context = retrieve_context(test_query_english)
    for i, doc in enumerate(context):
        print(f"  Result {i+1}: {doc[:100]}...")
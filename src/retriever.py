import chromadb
from sentence_transformers import SentenceTransformer
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_context(
    query: str, 
    db_path: str, 
    collection_name: str, 
    model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2',
    n_results: int = 3,
    threshold: float = 0.0
) -> List[str]:
    """
    Retrieves relevant context from the ChromaDB vector store based on a query.

    Args:
        query: The user's query.
        db_path: The path to the ChromaDB directory.
        collection_name: The name of the collection to query.
        model_name: The name of the Sentence Transformers model to use.
        n_results: The number of documents to retrieve.
        threshold: The maximum distance score for a document to be considered relevant.
                   ChromaDB uses distance metrics (like L2 squared), so a lower score is better.
                   A value of 0.0 means the threshold is disabled.

    Returns:
        A list of relevant document texts.
    """
    try:
        logging.info(f"Initializing ChromaDB client at path: {db_path}")
        client = chromadb.PersistentClient(path=db_path)

        # Check if the collection exists
        if collection_name not in [c.name for c in client.list_collections()]:
            logging.error(f"Collection '{collection_name}' does not exist in the database at {db_path}.")
            return []

        logging.info(f"Getting collection: {collection_name}")
        collection = client.get_collection(name=collection_name)

        logging.info(f"Loading sentence transformer model: {model_name}")
        model = SentenceTransformer(model_name)

        logging.info(f"Generating embedding for query: '{query}'")
        query_embedding = model.encode(query).tolist()

        logging.info(f"Querying collection for {n_results} results...")
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]

        if not documents:
            logging.warning("No documents found in the query result.")
            return []

        # If a threshold is set, filter the results
        if threshold > 0.0:
            relevant_docs = [
                doc for doc, dist in zip(documents, distances) if dist <= threshold
            ]
            logging.info(f"Found {len(relevant_docs)} documents meeting the distance threshold of {threshold}")
            return relevant_docs
        
        # If no threshold, return the top n_results
        logging.info(f"Retrieved {len(documents)} relevant documents (no threshold).")
        return documents

    except Exception as e:
        logging.error(f"An error occurred during context retrieval: {e}")
        return []

if __name__ == '__main__':
    # This is for example usage and simple testing.
    # Ensure the vector store is built by running `build_vector_store.py` first.
    DB_PATH = 'chroma_db'
    COLLECTION_NAME = 'medical_faqs'
    
    test_query_english = "What are the symptoms of the flu?"
    print(f"\n--- Querying with: '{test_query_english}' ---")
    context = retrieve_context(test_query_english, DB_PATH, COLLECTION_NAME)
    for i, doc in enumerate(context):
        print(f"  Result {i+1}: {doc[:100]}...")

    test_query_spanish = "qué es la fiebre"
    print(f"\n--- Querying with: '{test_query_spanish}' ---")
    context_spanish = retrieve_context(test_query_spanish, DB_PATH, COLLECTION_NAME)
    for i, doc in enumerate(context_spanish):
        print(f"  Result {i+1}: {doc[:100]}...")

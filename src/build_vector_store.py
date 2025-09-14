import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
import logging
from tqdm import tqdm
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.config import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL_NAME, DATA_PATH

def create_vector_store(
    docs: List[Dict[str, str]], 
    collection_name: str, 
    model_name: str = EMBEDDING_MODEL_NAME,
    db_path: Optional[str] = None, 
    client: Optional[chromadb.Client] = None
):
    """
    Creates or updates a ChromaDB vector store from a list of documents.

    Args:
        docs: A list of documents to add to the collection.
        collection_name: The name of the collection.
        db_path: Path for the persistent database. If None, an in-memory client must be provided.
        client: An optional chromadb.Client instance. If not provided, a persistent client
                will be created using db_path.
        model_name: The Sentence Transformers model to use for embeddings.
    """
    if client is None:
        if db_path:
            os.makedirs(db_path, exist_ok=True)
            logging.info(f"Initializing ChromaDB persistent client at path: {db_path}")
            client = chromadb.PersistentClient(path=db_path)
        else:
            raise ValueError("Either a 'db_path' for a persistent client or a 'client' instance must be provided.")
    else:
        logging.info("Using provided ChromaDB client.")

    try:
        client.delete_collection(name=collection_name)
        logging.info(f"Successfully deleted existing collection: '{collection_name}'")
    except Exception:
        logging.info(f"Collection '{collection_name}' did not exist, creating new one.")
        pass # Collection doesn't exist, so we don't need to delete it

    collection = client.create_collection(name=collection_name)
    logging.info(f"Successfully created new empty collection: '{collection_name}'")

    if not docs:
        logging.warning("Document list is empty. No new data will be added.")
        return

    logging.info(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    batch_size = 100
    logging.info(f"Processing {len(docs)} documents in batches of {batch_size}...")

    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents"):
        batch_docs = docs[i:i+batch_size]
        texts_to_embed = [doc['text'] for doc in batch_docs if doc and doc.get('text')]
        metadatas = [{"source_id": doc['source_id']} for doc in batch_docs if doc and doc.get('text')]
        
        if not texts_to_embed:
            continue

        # --- DEBUGGING ---
        if i == 0: # Check the first batch
            logging.info(f"DEBUG: Sample metadata from first batch: {metadatas[:5]}")
        # --- END DEBUGGING ---

        embeddings = model.encode(texts_to_embed, show_progress_bar=False).tolist()
        ids = [f"doc_{collection.count() + j}" for j in range(len(texts_to_embed))]
        
        collection.add(embeddings=embeddings, documents=texts_to_embed, metadatas=metadatas, ids=ids)

    logging.info(f"Vector store update complete. Collection '{collection_name}' now contains {collection.count()} documents.")

if __name__ == '__main__':
    from data_loader import load_data
    
    logging.info(f"Loading data from {DATA_PATH}...")
    documents = load_data(DATA_PATH)
    
    # The main script will use the persistent client with config values
    create_vector_store(
        docs=documents, 
        collection_name=COLLECTION_NAME, 
        model_name=EMBEDDING_MODEL_NAME,
        db_path=DB_PATH
    )
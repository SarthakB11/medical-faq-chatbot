import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import os

def create_vector_store(docs: List[Dict[str, str]], db_path: str, collection_name: str, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
    """
    Creates a ChromaDB vector store from a list of documents.

    Args:
        docs: A list of documents, where each document is a dictionary
              expected to have a 'text' key.
        db_path: The path to the directory where the database will be stored.
        collection_name: The name of the collection to create in the database.
        model_name: The name of the Sentence Transformers model to use for embeddings.
    """
    # Ensure the database directory exists
    os.makedirs(db_path, exist_ok=True)
    
    logging.info(f"Initializing ChromaDB client at path: {db_path}")
    client = chromadb.PersistentClient(path=db_path)

    # Delete the collection if it already exists to ensure a fresh start
    try:
        if collection_name in [c.name for c in client.list_collections()]:
            logging.info(f"Collection '{collection_name}' already exists. Deleting it.")
            client.delete_collection(name=collection_name)
    except Exception as e:
        logging.warning(f"Could not check for or delete existing collection. This might be an issue on a fresh run. Error: {e}")


    logging.info(f"Getting or creating collection: {collection_name}")
    collection = client.get_or_create_collection(name=collection_name)

    if not docs:
        logging.warning("Document list is empty. The collection is created but contains no data.")
        return

    logging.info(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Process documents in batches for efficiency
    batch_size = 100
    logging.info(f"Processing {len(docs)} documents in batches of {batch_size}...")

    for i in tqdm(range(0, len(docs), batch_size), desc="Embedding documents"):
        batch_docs = docs[i:i+batch_size]
        
        # Filter out any potential None or empty documents
        texts_to_embed = [doc['text'] for doc in batch_docs if doc and 'text' in doc and doc['text']]
        if not texts_to_embed:
            continue

        # Generate embeddings
        embeddings = model.encode(texts_to_embed, show_progress_bar=False).tolist()
        
        # Create unique IDs for each document
        ids = [f"doc_{i+j}" for j in range(len(texts_to_embed))]
        
        # Add to the collection
        collection.add(
            embeddings=embeddings,
            documents=texts_to_embed,
            ids=ids
        )

    logging.info(f"Vector store creation complete. Collection '{collection_name}' contains {collection.count()} documents.")

if __name__ == '__main__':
    from data_loader import load_data
    
    # Configuration
    DATA_PATH = 'data/dummy_medical_data.csv'
    DB_PATH = 'chroma_db'
    COLLECTION_NAME = 'medical_faqs'

    # Load data
    logging.info(f"Loading data from {DATA_PATH}...")
    documents = load_data(DATA_PATH)
    
    # Create the vector store
    create_vector_store(documents, DB_PATH, COLLECTION_NAME)

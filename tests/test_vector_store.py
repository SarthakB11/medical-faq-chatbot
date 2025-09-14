import unittest
import chromadb
from src.build_vector_store import create_vector_store

class TestVectorStore(unittest.TestCase):

    def setUp(self):
        """Set up an in-memory ChromaDB client for each test."""
        self.client = chromadb.Client()
        self.collection_name = "test_collection"

    def test_create_vector_store_with_documents(self):
        """Test creating a vector store with documents using an in-memory client."""
        test_docs = [
            {"text": "This is a test document about ChromaDB.", "source_id": "test-1"},
            {"text": "Sentence transformers are great for embeddings.", "source_id": "test-2"},
        ]
        
        create_vector_store(
            docs=test_docs, 
            collection_name=self.collection_name, 
            client=self.client,
            model_name='paraphrase-multilingual-MiniLM-L12-v2' # Use a known model for testing
        )
        
        collection = self.client.get_collection(self.collection_name)
        self.assertEqual(collection.count(), len(test_docs))

    def test_create_vector_store_empty_docs(self):
        """Test creating a vector store with an empty list of documents."""
        create_vector_store(
            docs=[], 
            collection_name=self.collection_name, 
            client=self.client,
            model_name='paraphrase-multilingual-MiniLM-L12-v2'
        )
        
        collection = self.client.get_collection(self.collection_name)
        self.assertEqual(collection.count(), 0)

if __name__ == '__main__':
    unittest.main()

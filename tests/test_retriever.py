import unittest
import chromadb
from src.build_vector_store import create_vector_store
from src.retriever import retrieve_context
from src.config import EMBEDDING_MODEL_NAME

class TestRetriever(unittest.TestCase):

    def test_retrieve_context_english_query(self):
        """Test retrieving context with an English query."""
        client = chromadb.Client()
        collection_name = "test_english_query"
        test_docs = [{"text": "The flu is a contagious respiratory illness.", "source_id": "test-flu"}]
        
        create_vector_store(
            docs=test_docs, 
            collection_name=collection_name, 
            client=client,
            model_name=EMBEDDING_MODEL_NAME
        )
        
        # Disable the threshold to ensure the top result is always returned
        retrieved_docs = retrieve_context(
            "What is the flu?", 
            collection_name, 
            client=client, 
            threshold=0.0
        )
        
        self.assertGreater(len(retrieved_docs), 0)
        self.assertIn("contagious", retrieved_docs[0]['text'])
        self.assertIn("metadata", retrieved_docs[0])

    def test_retrieve_context_spanish_query(self):
        """Test retrieving context with a Spanish query."""
        client = chromadb.Client()
        collection_name = "test_spanish_query"
        test_docs = [{"text": "La fiebre es un síntoma común.", "source_id": "test-fiebre"}]

        create_vector_store(
            docs=test_docs, 
            collection_name=collection_name, 
            client=client,
            model_name=EMBEDDING_MODEL_NAME
        )

        # Disable the threshold
        retrieved_docs = retrieve_context(
            "qué es la fiebre", 
            collection_name, 
            client=client, 
            threshold=0.0
        )
        
        self.assertGreater(len(retrieved_docs), 0)
        self.assertIn("síntoma", retrieved_docs[0]['text'])
        self.assertIn("metadata", retrieved_docs[0])

    def test_retrieve_context_no_match(self):
        """Test that an empty list is returned when the threshold is strict."""
        client = chromadb.Client()
        collection_name = "test_no_match"
        test_docs = [{"text": "This is a test document about medicine.", "source_id": "test-medicine"}]

        create_vector_store(
            docs=test_docs, 
            collection_name=collection_name, 
            client=client,
            model_name=EMBEDDING_MODEL_NAME
        )

        # Use a very strict threshold to ensure no match
        retrieved_docs = retrieve_context(
            "query about astrophysics", 
            collection_name, 
            client=client, 
            threshold=0.1
        )
        
        self.assertEqual(len(retrieved_docs), 0)

if __name__ == '__main__':
    unittest.main()
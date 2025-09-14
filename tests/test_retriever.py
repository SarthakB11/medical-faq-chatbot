import unittest
import chromadb
from src.build_vector_store import create_vector_store
from src.retriever import retrieve_context

class TestRetriever(unittest.TestCase):

    def setUp(self):
        """Set up an in-memory client and populate it with test data."""
        self.client = chromadb.Client()
        self.collection_name = "test_retriever_collection"
        self.test_docs = [
            {"text": "The flu is a contagious respiratory illness caused by influenza viruses."},
            {"text": "Fever is a common symptom of many illnesses."},
            {"text": "La fiebre es un síntoma común de muchas enfermedades."} # Spanish
        ]
        
        create_vector_store(
            docs=self.test_docs,
            collection_name=self.collection_name,
            client=self.client
        )

    def test_retrieve_context_english_query(self):
        """Test retrieving context with an English query."""
        query = "What is the flu?"
        retrieved_docs = retrieve_context(query, self.collection_name, client=self.client)
        
        self.assertIsNotNone(retrieved_docs)
        self.assertGreater(len(retrieved_docs), 0)
        self.assertIn("influenza viruses", retrieved_docs[0])

    def test_retrieve_context_spanish_query(self):
        """Test retrieving context with a Spanish query."""
        query = "Cuales son los sintomas de la fiebre?" # "What are the symptoms of fever?"
        retrieved_docs = retrieve_context(query, self.collection_name, client=self.client)
        
        self.assertIsNotNone(retrieved_docs)
        self.assertGreater(len(retrieved_docs), 0)
        # Should retrieve the Spanish or English text about fever
        self.assertTrue(
            "Fever is a common symptom" in retrieved_docs[0] or 
            "La fiebre es un síntoma" in retrieved_docs[0]
        )

    def test_retrieve_context_no_match(self):
        """Test that an empty list is returned when no relevant context is found."""
        query = "Information about astrophysics"
        retrieved_docs = retrieve_context(
            query, self.collection_name, client=self.client, n_results=1, threshold=0.2
        )
        self.assertEqual(len(retrieved_docs), 0)

if __name__ == '__main__':
    unittest.main()
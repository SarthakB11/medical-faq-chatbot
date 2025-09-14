import unittest
import os
import shutil
from src.build_vector_store import create_vector_store
from src.retriever import retrieve_context

class TestRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Create the vector store once for all tests in this class."""
        cls.db_path = "test_retriever_db"
        cls.collection_name = "test_retriever_collection"
        cls.test_docs = [
            {"text": "The flu is a contagious respiratory illness caused by influenza viruses."},
            {"text": "Fever is a common symptom of many illnesses."},
            {"text": "qué es la fiebre"} # Spanish for "what is fever"
        ]
        # Ensure the directory is clean before starting
        if os.path.exists(cls.db_path):
            shutil.rmtree(cls.db_path)
        
        create_vector_store(cls.test_docs, cls.db_path, cls.collection_name)

    @classmethod
    def tearDownClass(cls):
        """Remove the temporary vector store after all tests."""
        if os.path.exists(cls.db_path):
            shutil.rmtree(cls.db_path)

    def test_retrieve_context_english_query(self):
        """Test retrieving context with an English query."""
        query = "What is the flu?"
        retrieved_docs = retrieve_context(query, self.db_path, self.collection_name)
        
        self.assertIsNotNone(retrieved_docs)
        self.assertGreater(len(retrieved_docs), 0)
        self.assertIn("influenza viruses", retrieved_docs[0])

    def test_retrieve_context_spanish_query(self):
        """Test retrieving context with a Spanish query."""
        query = "Cuales son los sintomas de la fiebre?" # "What are the symptoms of fever?"
        retrieved_docs = retrieve_context(query, self.db_path, self.collection_name)
        
        self.assertIsNotNone(retrieved_docs)
        self.assertGreater(len(retrieved_docs), 0)
        self.assertIn("Fever is a common symptom", retrieved_docs[0])

    def test_retrieve_context_no_match(self):
        """Test that an empty list is returned when no relevant context is found."""
        query = "Information about astrophysics"
        # Using a tight threshold to ensure no match
        retrieved_docs = retrieve_context(query, self.db_path, self.collection_name, n_results=1, threshold=0.2)
        
        self.assertEqual(len(retrieved_docs), 0)

if __name__ == '__main__':
    unittest.main()

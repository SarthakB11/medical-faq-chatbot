import unittest
import os
import pandas as pd
from src.data_loader import load_data

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        """Set up a dummy CSV file for testing."""
        self.test_csv_path = 'test_data.csv'
        data = {
            'Question': ['Q1', 'Q2'],
            'Answer': ['A1', 'A2']
        }
        pd.DataFrame(data).to_csv(self.test_csv_path, index=False)

    def tearDown(self):
        """Remove the dummy CSV file after tests."""
        if os.path.exists(self.test_csv_path):
            os.remove(self.test_csv_path)

    def test_load_data_success(self):
        """Test that data is loaded correctly from a CSV file."""
        faqs = load_data(self.test_csv_path)
        self.assertIsInstance(faqs, list)
        self.assertEqual(len(faqs), 2)
        self.assertIn('Question', faqs[0])
        self.assertIn('Answer', faqs[0])
        self.assertEqual(faqs[0]['Question'], 'Q1')
        self.assertEqual(faqs[0]['Answer'], 'A1')

    def test_load_data_file_not_found(self):
        """Test that a FileNotFoundError is raised for a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            load_data('non_existent_file.csv')

    def test_load_data_empty_file(self):
        """Test that an empty list is returned for an empty CSV file."""
        empty_csv_path = 'empty_test_data.csv'
        with open(empty_csv_path, 'w') as f:
            f.write('Question,Answer\n') # Just the header
        
        faqs = load_data(empty_csv_path)
        self.assertEqual(len(faqs), 0)
        
        os.remove(empty_csv_path)

if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.mock import patch, MagicMock
from src.answer_generator import generate_answer

class TestAnswerGenerator(unittest.TestCase):

    @patch('src.answer_generator.llm')
    def test_generate_answer_english(self, mock_llm):
        """Test that the prompt is formatted correctly for an English query."""
        # Configure the mock LLM to return a fake response
        mock_llm.generate.return_value = "This is a mock answer about the flu."

        query = "What is the flu?"
        context = ["The flu is a contagious respiratory illness caused by influenza viruses."]
        
        answer = generate_answer(query, context)

        self.assertEqual(answer, "This is a mock answer about the flu.")
        
        # Check that our mock LLM's generate method was called once
        mock_llm.generate.assert_called_once()
        
        # Check that the prompt passed to the generate method was correct
        prompt = mock_llm.generate.call_args[0][0]
        self.assertIn("You are a helpful medical assistant", prompt)
        self.assertIn(query, prompt)
        self.assertIn(context[0], prompt)
        self.assertIn("Answer in English", prompt)

    @patch('src.answer_generator.llm')
    def test_generate_answer_spanish(self, mock_llm):
        """Test that the prompt is formatted correctly for a Spanish query."""
        mock_llm.generate.return_value = "Esta es una respuesta simulada sobre la fiebre."

        query = "qué es la fiebre"
        context = ["Fever is a common symptom of many illnesses."]
        
        answer = generate_answer(query, context, language="Spanish")

        self.assertEqual(answer, "Esta es una respuesta simulada sobre la fiebre.")
        mock_llm.generate.assert_called_once()
        
        prompt = mock_llm.generate.call_args[0][0]
        self.assertIn("Answer in Spanish", prompt)

    @patch('src.answer_generator.llm')
    def test_generate_answer_no_context(self, mock_llm):
        """Test the prompt format when no context is provided."""
        mock_llm.generate.return_value = "I cannot answer that."

        query = "What is the meaning of life?"
        context = []
        
        answer = generate_answer(query, context)

        self.assertEqual(answer, "I cannot answer that.")
        
        prompt = mock_llm.generate.call_args[0][0]
        self.assertIn(query, prompt)
        self.assertIn("No relevant context was found", prompt)

if __name__ == '__main__':
    unittest.main()

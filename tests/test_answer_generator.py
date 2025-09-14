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
        context = [
            {"text": "The flu is a contagious respiratory illness.", "metadata": {"source_id": "FAQ-1"}},
            {"text": "Symptoms include fever and cough.", "metadata": {"source_id": "FAQ-2"}}
        ]
        
        answer = generate_answer(query, context)

        self.assertEqual(answer, "This is a mock answer about the flu.")
        
        mock_llm.generate.assert_called_once()
        
        prompt = mock_llm.generate.call_args[0][0]
        self.assertIn("You are a helpful medical assistant", prompt)
        self.assertIn(query, prompt)
        self.assertIn("Source: [FAQ-1]", prompt)
        self.assertIn("Content: The flu is a contagious respiratory illness.", prompt)
        self.assertIn("Source: [FAQ-2]", prompt)
        self.assertIn("Content: Symptoms include fever and cough.", prompt)
        self.assertIn("you MUST cite the sources you used", prompt)

    @patch('src.answer_generator.llm')
    def test_generate_answer_spanish(self, mock_llm):
        """Test that the prompt is formatted correctly for a Spanish query."""
        mock_llm.generate.return_value = "Esta es una respuesta simulada sobre la fiebre."

        query = "qué es la fiebre"
        context = [{"text": "Fever is a common symptom.", "metadata": {"source_id": "FAQ-3"}}]
        
        answer = generate_answer(query, context, language="Spanish")

        self.assertEqual(answer, "Esta es una respuesta simulada sobre la fiebre.")
        mock_llm.generate.assert_called_once()
        
        prompt = mock_llm.generate.call_args[0][0]
        self.assertIn("Answer in Spanish", prompt)
        self.assertIn("Source: [FAQ-3]", prompt)

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

import unittest
from unittest.mock import patch, MagicMock
from src.answer_generator import generate_answer

class TestAnswerGenerator(unittest.TestCase):

    @patch('src.answer_generator.genai.GenerativeModel')
    def test_generate_answer_english(self, MockGenerativeModel):
        """Test that the answer generation formats the prompt correctly for an English query."""
        # Configure the mock to return a fake response
        mock_instance = MockGenerativeModel.return_value
        mock_instance.generate_content.return_value.text = "This is a mock answer about the flu."

        query = "What is the flu?"
        context = ["The flu is a contagious respiratory illness caused by influenza viruses."]
        
        answer = generate_answer(query, context)

        self.assertEqual(answer, "This is a mock answer about the flu.")
        mock_instance.generate_content.assert_called_once()
        
        # Check that the prompt was constructed correctly
        prompt = mock_instance.generate_content.call_args[0][0]
        self.assertIn("You are a helpful medical assistant", prompt)
        self.assertIn(query, prompt)
        self.assertIn(context[0], prompt)
        self.assertIn("Answer in English", prompt)

    @patch('src.answer_generator.genai.GenerativeModel')
    def test_generate_answer_spanish(self, MockGenerativeModel):
        """Test that the answer generation formats the prompt correctly for a Spanish query."""
        mock_instance = MockGenerativeModel.return_value
        mock_instance.generate_content.return_value.text = "Esta es una respuesta simulada sobre la fiebre."

        query = "qué es la fiebre"
        context = ["Fever is a common symptom of many illnesses."]
        
        answer = generate_answer(query, context, language="Spanish")

        self.assertEqual(answer, "Esta es una respuesta simulada sobre la fiebre.")
        mock_instance.generate_content.assert_called_once()
        
        prompt = mock_instance.generate_content.call_args[0][0]
        self.assertIn("Answer in Spanish", prompt)

    @patch('src.answer_generator.genai.GenerativeModel')
    def test_generate_answer_no_context(self, MockGenerativeModel):
        """Test that the function can handle cases with no context."""
        mock_instance = MockGenerativeModel.return_value
        mock_instance.generate_content.return_value.text = "I cannot answer that question without more information."

        query = "What is the meaning of life?"
        context = []
        
        answer = generate_answer(query, context)

        self.assertEqual(answer, "I cannot answer that question without more information.")
        
        prompt = mock_instance.generate_content.call_args[0][0]
        self.assertIn(query, prompt)
        self.assertIn("No relevant context was found", prompt)

if __name__ == '__main__':
    unittest.main()
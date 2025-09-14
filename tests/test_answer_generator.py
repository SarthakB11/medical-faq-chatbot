import unittest
from unittest.mock import patch, MagicMock
from src.answer_generator import generate_answer

class TestAnswerGenerator(unittest.TestCase):

    @patch('src.answer_generator.openai.ChatCompletion.create')
    def test_generate_answer_english(self, mock_create):
        """Test that the answer generation formats the prompt correctly for an English query."""
        # Configure the mock to return a fake response
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "This is a mock answer about the flu."
        mock_create.return_value = mock_response

        query = "What is the flu?"
        context = ["The flu is a contagious respiratory illness caused by influenza viruses."]
        
        answer = generate_answer(query, context)

        # Check that the function returned the content of the mock response
        self.assertEqual(answer, "This is a mock answer about the flu.")

        # Check that the OpenAI API was called
        mock_create.assert_called_once()
        
        # Check that the prompt was constructed correctly
        _, kwargs = mock_create.call_args
        messages = kwargs['messages']
        
        system_prompt = messages[0]['content']
        user_prompt = messages[1]['content']

        self.assertIn("You are a helpful medical assistant", system_prompt)
        self.assertIn(query, user_prompt)
        self.assertIn(context[0], user_prompt)
        self.assertIn("Answer in English", system_prompt)

    @patch('src.answer_generator.openai.ChatCompletion.create')
    def test_generate_answer_spanish(self, mock_create):
        """Test that the answer generation formats the prompt correctly for a Spanish query."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Esta es una respuesta simulada sobre la fiebre."
        mock_create.return_value = mock_response

        query = "qué es la fiebre"
        context = ["Fever is a common symptom of many illnesses."]
        
        answer = generate_answer(query, context, language="Spanish")

        self.assertEqual(answer, "Esta es una respuesta simulada sobre la fiebre.")
        mock_create.assert_called_once()
        
        _, kwargs = mock_create.call_args
        messages = kwargs['messages']
        system_prompt = messages[0]['content']

        self.assertIn("Answer in Spanish", system_prompt)

    @patch('src.answer_generator.openai.ChatCompletion.create')
    def test_generate_answer_no_context(self, mock_create):
        """Test that the function can handle cases with no context."""
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "I cannot answer that question without more information."
        mock_create.return_value = mock_response

        query = "What is the meaning of life?"
        context = []
        
        answer = generate_answer(query, context)

        self.assertEqual(answer, "I cannot answer that question without more information.")
        
        _, kwargs = mock_create.call_args
        user_prompt = kwargs['messages'][1]['content']
        
        # The prompt should still contain the query but indicate no context was found
        self.assertIn(query, user_prompt)
        self.assertIn("No relevant context was found", user_prompt)

if __name__ == '__main__':
    unittest.main()

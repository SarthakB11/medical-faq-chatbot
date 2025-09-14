# src/llm.py

from abc import ABC, abstractmethod
from typing import List
import google.generativeai as genai
import logging
from src.config import LLM_MODEL_NAME

class LanguageModel(ABC):
    """Abstract base class for a language model."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generates a response based on a given prompt."""
        pass

class GeminiModel(LanguageModel):
    """Implementation of the LanguageModel class for Google's Gemini."""

    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        try:
            self.model = genai.GenerativeModel(self.model_name)
            logging.info(f"Successfully initialized Gemini model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model '{self.model_name}': {e}")
            self.model = None

    def generate(self, prompt: str) -> str:
        """
        Generates a response using the configured Gemini model.
        """
        if not self.model:
            return "Error: Gemini model is not initialized. Please check your API key and model name."

        try:
            logging.info(f"Calling Gemini API with model: {self.model_name}...")
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            logging.info("Successfully received response from Gemini.")
            return answer
        except Exception as e:
            logging.error(f"An error occurred while calling the Gemini API: {e}")
            return "I'm sorry, but I encountered an error while trying to generate an answer."

# --- Factory Function ---
def get_language_model() -> LanguageModel:
    """
    Factory function to get the currently configured language model.
    This makes it easy to switch between models in the future.
    """
    # For now, we only have Gemini, but you could add logic here to
    # instantiate different models based on a config setting.
    return GeminiModel()

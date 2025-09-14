# src/llm.py

from abc import ABC, abstractmethod
from typing import List, Iterator
import google.generativeai as genai
import logging
import os
from src.config import LLM_MODEL_NAME

class LanguageModel(ABC):
    """Abstract base class for a language model."""
    @abstractmethod
    def generate(self, prompt: str) -> str:
        pass

    @abstractmethod
    def generate_stream(self, prompt: str) -> Iterator[str]:
        pass

class GeminiModel(LanguageModel):
    """Implementation of the LanguageModel class for Google's Gemini."""
    def __init__(self, model_name: str = LLM_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self._configure_api_key()

    def _configure_api_key(self):
        """Configures the Gemini API key from environment variables."""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logging.error("GEMINI_API_KEY not found in environment variables.")
            return
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logging.info(f"Successfully initialized Gemini model: {self.model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model '{self.model_name}': {e}")

    def generate(self, prompt: str) -> str:
        """Generates a complete response."""
        if not self.model:
            return "Error: Gemini model is not initialized. Check API key."
        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Gemini API call error: {e}")
            return "An error occurred while generating an answer."

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Generates a response as a stream of chunks."""
        if not self.model:
            yield "Error: Gemini model is not initialized. Check API key."
            return
        try:
            response_stream = self.model.generate_content(prompt, stream=True)
            for chunk in response_stream:
                yield chunk.text
        except Exception as e:
            logging.error(f"Gemini API stream error: {e}")
            yield "An error occurred while generating an answer."

def get_language_model() -> LanguageModel:
    """Factory function to get the currently configured language model."""
    return GeminiModel()

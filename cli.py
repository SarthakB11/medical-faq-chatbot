import argparse
import os
from src.retriever import retrieve_context
from src.answer_generator import generate_answer
from src.config import DB_PATH, VECTOR_STORE_EXISTS
import logging
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent detection results
DetectorFactory.seed = 0

# The answer_generator module now handles loading the .env file.

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_language_name(lang_code):
    """Converts a language code (e.g., 'en') to its full name (e.g., 'English')."""
    # This is a simple mapping. A more robust solution might use a library.
    lang_map = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
        "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic"
    }
    return lang_map.get(lang_code, "English") # Default to English

def main():
    """
    Main function for the command-line interface of the Medical FAQ Chatbot.
    """
    if not VECTOR_STORE_EXISTS:
        logging.error(f"Vector store not found. Please run `build_vector_store.py`.")
        return

    print("--- Medical FAQ Chatbot CLI ---")
    print("Ask a question, or type 'exit' to quit.")
    
    history = []
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        try:
            lang_code = detect(query)
            language = get_language_name(lang_code)
            logging.info(f"Detected language: {language} ({lang_code})")
        except LangDetectException:
            logging.warning("Could not detect language. Defaulting to English.")
            language = "English"

        logging.info("1. Retrieving context...")
        retrieved_docs = retrieve_context(query)

        if not retrieved_docs:
            print("\nBot: I could not find any relevant information to answer your question.")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "I could not find any relevant information."})
            continue

        logging.info("2. Generating answer...")
        answer = generate_answer(query, retrieved_docs, history=history, language=language)

        print(f"\nBot: {answer}")
        
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()

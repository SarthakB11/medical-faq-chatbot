import argparse
import os
from src.retriever import retrieve_context
from src.answer_generator import generate_answer
from src.config import DB_PATH
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
    lang_map = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
        "zh-cn": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic"
    }
    return lang_map.get(lang_code, "English")

def main():
    """
    Main function for the command-line interface of the Medical FAQ Chatbot.
    """
    parser = argparse.ArgumentParser(description="RAG-based Medical FAQ Chatbot CLI")
    parser.add_argument("query", type=str, help="Your medical question.")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        logging.error(f"Vector store not found. Please run `build_vector_store.py`.")
        return

    query = args.query
    
    try:
        lang_code = detect(query)
        language = get_language_name(lang_code)
    except LangDetectException:
        language = "English"

    logging.info(f"Received query: '{query}' (Language: {language})")

    retrieved_docs = retrieve_context(query, threshold=0.0)

    if not retrieved_docs:
        print("\nI could not find any relevant information to answer your question.")
        return

    answer = generate_answer(query, retrieved_docs, language=language)

    print("\n--- Answer ---")
    print(answer)
    print("---------------")

if __name__ == "__main__":
    main()
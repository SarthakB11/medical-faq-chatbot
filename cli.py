import argparse
import os
from src.retriever import retrieve_context
from src.answer_generator import generate_answer, rewrite_query
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
    if not os.path.exists(DB_PATH):
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
        except LangDetectException:
            language = "English"

        logging.info(f"Received query: '{query}' (Language: {language})")

        # 1. Rewrite the query
        rewritten = rewrite_query(query, history[-10:])
        logging.info(f"Rewritten query: '{rewritten}'")

        # 2. Retrieve context with the rewritten query
        retrieved_docs = retrieve_context(rewritten, threshold=0.0)

        if not retrieved_docs:
            print("\nBot: I could not find any relevant information to answer your question.")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "I could not find any relevant information."})
            continue

        # 3. Generate the answer with the original query
        answer = generate_answer(query, retrieved_docs, language=language)

        print(f"\nBot: {answer}")
        
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
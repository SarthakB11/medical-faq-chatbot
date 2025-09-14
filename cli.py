import argparse
import os
from src.retriever import retrieve_context
from src.answer_generator import generate_answer
from src.config import DB_PATH, VECTOR_STORE_EXISTS
import logging

# The answer_generator module now handles loading the .env file.

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function for the command-line interface of the Medical FAQ Chatbot.
    """
    parser = argparse.ArgumentParser(description="RAG-based Medical FAQ Chatbot CLI")
    parser.add_argument("query", type=str, help="Your medical question.")
    parser.add_argument(
        "--lang", 
        type=str, 
        default="English", 
        help="The language for the answer (e.g., 'Spanish', 'French')."
    )
    args = parser.parse_args()

    if not VECTOR_STORE_EXISTS:
        logging.error(
            "The vector store database was not found. "
            f"Please run `build_vector_store.py` to create it at: {DB_PATH}"
        )
        return

    query = args.query
    language = args.lang

    logging.info(f"Received query: '{query}'")

    # --- RAG Pipeline ---
    logging.info("1. Retrieving context from the knowledge base...")
    retrieved_docs = retrieve_context(query)

    if not retrieved_docs:
        print("\nI could not find any relevant information in the knowledge base to answer your question.")
        return

    logging.info("2. Generating an answer based on the context...")
    answer = generate_answer(query, retrieved_docs, language=language)

    # --- Display Answer ---
    print("\n--- Answer ---")
    print(answer)
    print("--------------- ")

if __name__ == "__main__":
    main()

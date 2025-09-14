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
    if not VECTOR_STORE_EXISTS:
        logging.error(
            "The vector store database was not found. "
            f"Please run `build_vector_store.py` to create it at: {DB_PATH}"
        )
        return

    print("--- Medical FAQ Chatbot CLI ---")
    print("Ask a question, or type 'exit' to quit.")
    
    history = []
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            break

        logging.info(f"Received query: '{query}'")

        # --- RAG Pipeline ---
        logging.info("1. Retrieving context...")
        retrieved_docs = retrieve_context(query)

        if not retrieved_docs:
            print("\nBot: I could not find any relevant information to answer your question.")
            history.append({"role": "user", "content": query})
            history.append({"role": "assistant", "content": "I could not find any relevant information."})
            continue

        logging.info("2. Generating answer...")
        answer = generate_answer(query, retrieved_docs, history=history)

        # --- Display Answer ---
        print(f"\nBot: {answer}")
        
        # Update history
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
import google.generativeai as genai
import os
from typing import List
import logging
from dotenv import load_dotenv

# --- Load Environment Variables ---
# Find the project root by going up one level from this script's directory (src)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configure the Gemini API key
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    logging.error("GEMINI_API_KEY environment variable not found. Please ensure it is set.")
else:
    genai.configure(api_key=api_key)

def generate_answer(query: str, context: List[str], language: str = "English") -> str:
    """
    Generates an answer to a query using the Gemini API, based on the provided context.
    """
    if not api_key:
        return "Error: Gemini API key is not configured. Please check your environment variables."

    context_str = "\n\n".join(context)
    
    if not context:
        prompt = f"""You are a helpful medical assistant. The user has asked the following question: '{query}'. No relevant context was found in the knowledge base. Please inform the user that you cannot answer this question with the available information. Answer in {language}."""
    else:
        prompt = f"""You are a helpful medical assistant. Your purpose is to answer medical questions based on the context provided. Be concise, accurate, and easy to understand. If the context does not contain the answer, say that you cannot answer the question based on the provided information. Do not use any information outside of the given context. Answer in {language}.

Based on the following context, please answer the user's question.

Context:
---
{context_str}
---

Question: {query}
"""

    try:
        logging.info("Calling Gemini API...")
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        logging.info("Successfully received response from Gemini.")
        return answer
    except Exception as e:
        logging.error(f"An error occurred while calling the Gemini API: {e}")
        return "I'm sorry, but I encountered an error while trying to generate an answer. Please try again later."

if __name__ == '__main__':
    test_query = "What are the symptoms of the flu?"
    test_context = ["The flu is a contagious respiratory illness..."]
    
    if not api_key:
        print("Cannot run example because GEMINI_API_KEY is not set.")
    else:
        print(f"--- Generating answer for: '{test_query}' ---")
        print(generate_answer(test_query, test_context))

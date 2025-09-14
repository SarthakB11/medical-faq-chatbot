import os
from typing import List, Dict
import logging
from dotenv import load_dotenv
from src.llm import get_language_model

# --- Load Environment Variables ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Initialize Language Model ---
# The factory function handles the specific implementation (e.g., Gemini)
llm = get_language_model()

def generate_answer(query: str, context: List[Dict[str, any]], language: str = "English") -> str:
    """
    Constructs a prompt and generates an answer using the configured language model.

    Args:
        query: The user's query.
        context: A list of dictionaries, each containing a text chunk and its metadata.
        language: The language for the answer.

    Returns:
        The generated answer.
    """
    # Format the context with source citations
    context_parts = []
    for i, item in enumerate(context):
        text = item.get("text", "")
        source = item.get("metadata", {}).get("source_id", f"Source {i+1}")
        context_parts.append(f"Source: [{source}]\nContent: {text}")
    
    context_str = "\n\n".join(context_parts)
    
    if not context:
        prompt = f"""You are a helpful medical assistant. The user has asked the following question: '{query}'. No relevant context was found in the knowledge base. Please inform the user that you cannot answer this question with the available information. Answer in {language}."""
    else:
        prompt = f"""You are a helpful medical assistant. Your purpose is to answer medical questions based ONLY on the context provided below. Be concise, accurate, and easy to understand. After providing the answer, you MUST cite the sources you used in a 'Sources:' section, like this: 'Sources: [FAQ-123], [FAQ-456]'. Do not use any information outside of the given context. Answer in {language}.

Based on the following context, please answer the user's question.

Context:
---
{context_str}
---

Question: {query}
"""
    # The actual API call is now handled by the llm object
    return llm.generate(prompt)


if __name__ == '__main__':
    test_query = "What are the symptoms of the flu?"
    test_context = ["The flu is a contagious respiratory illness caused by influenza viruses..."]
    
    print(f"--- Generating answer for: '{test_query}' ---")
    print(generate_answer(test_query, test_context))
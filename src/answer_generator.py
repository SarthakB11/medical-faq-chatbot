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

def generate_answer(
    query: str, 
    context: List[Dict[str, any]], 
    history: List[Dict[str, str]] = [], 
    language: str = "English"
) -> str:
    """
    Constructs a prompt with conversation history and generates an answer.

    Args:
        query: The user's current query.
        context: A list of relevant context dictionaries.
        history: A list of previous user/assistant messages.
        language: The language for the answer.

    Returns:
        The generated answer.
    """
    # Format the conversation history for the prompt
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    # Format the context with source citations
    context_parts = []
    for i, item in enumerate(context):
        text = item.get("text", "")
        source = item.get("metadata", {}).get("source_id", f"Source {i+1}")
        context_parts.append(f"Source: [{source}]\nContent: {text}")
    
    context_str = "\n\n".join(context_parts)
    
    if not context:
        prompt = f"""You are a helpful medical assistant. The user has asked the following question: '{query}'. No relevant context was found in the knowledge base. Please inform the user that you cannot answer this question with the available information. Consider the conversation history for context.

Conversation History:
---
{history_str}
---

Question: {query}
Answer in {language}."""
    else:
        prompt = f"""You are a helpful medical assistant. Your purpose is to answer medical questions based ONLY on the context provided below. Consider the conversation history to understand follow-up questions. Be concise, accurate, and easy to understand. After providing the answer, you MUST cite the sources you used in a 'Sources:' section. Do not use any information outside of the given context. Answer in {language}.

Conversation History:
---
{history_str}
---

Context:
---
{context_str}
---

Question: {query}
"""
    return llm.generate(prompt)



if __name__ == '__main__':
    test_query = "What are the symptoms of the flu?"
    test_context = ["The flu is a contagious respiratory illness caused by influenza viruses..."]
    
    print(f"--- Generating answer for: '{test_query}' ---")
    print(generate_answer(test_query, test_context))
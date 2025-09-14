import os
from typing import List, Dict, Iterator
import logging
from dotenv import load_dotenv
from src.llm import get_language_model

# --- Load Environment Variables ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Initialize Language Model ---
llm = get_language_model()

def _construct_prompt(query: str, context: List[Dict[str, any]], history: List[Dict[str, str]], language: str) -> str:
    """Helper function to construct the full prompt."""
    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    context_parts = []
    for i, item in enumerate(context):
        text = item.get("text", "")
        source = item.get("metadata", {}).get("source_id", f"Source {i+1}")
        context_parts.append(f"Source: [{source}]\nContent: {text}")
    context_str = "\n\n".join(context_parts)
    
    if not context:
        return f"""You are a helpful medical assistant. The user has asked: '{query}'. No relevant context was found. Inform the user you cannot answer. Consider the conversation history for context.

Conversation History:
---
{history_str}
---

Question: {query}
Answer in {language}."""
    else:
        return f"""You are a helpful medical assistant. Answer the user's question based ONLY on the context below. Cite your sources after the answer. Consider the conversation history for follow-up questions. Answer in {language}.

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

def generate_answer(query: str, context: List[Dict[str, any]], history: List[Dict[str, str]] = [], language: str = "English") -> str:
    """Constructs a prompt and generates a complete answer."""
    prompt = _construct_prompt(query, context, history, language)
    return llm.generate(prompt)

def generate_answer_stream(query: str, context: List[Dict[str, any]], history: List[Dict[str, str]] = [], language: str = "English") -> Iterator[str]:
    """Constructs a prompt and generates a streamed answer."""
    prompt = _construct_prompt(query, context, history, language)
    return llm.generate_stream(prompt)

if __name__ == '__main__':
    test_query = "What are the symptoms of the flu?"
    test_context = [{"text": "The flu is a contagious respiratory illness...", "metadata": {"source_id": "FAQ-1"}}]
    
    print(f"--- Generating complete answer for: '{test_query}' ---")
    print(generate_answer(test_query, test_context))

    print(f"\n--- Generating streamed answer for: '{test_query}' ---")
    for chunk in generate_answer_stream(test_query, test_context):
        print(chunk, end="", flush=True)
    print()

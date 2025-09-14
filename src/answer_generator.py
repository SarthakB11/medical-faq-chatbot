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

def rewrite_query(query: str, history: List[Dict[str, str]]) -> str:
    """
    Rewrites a follow-up query into a standalone question using the conversation history.
    """
    if not history:
        return query

    history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
    
    prompt = f"""Based on the conversation history below, rewrite the user's final question to be a standalone question. If the final question is already standalone, just return it as is.

Conversation History:
---
{history_str}
---

User's Final Question: {query}

Rewritten Question:"""
    
    rewritten_query = llm.generate(prompt)
    # The logging of the rewritten query is useful for transparency, so we'll keep it.
    # A more advanced system might make this configurable.
    # logging.info(f"Rewritten query: '{rewritten_query}'")
    return rewritten_query

def _construct_prompt(query: str, context: List[Dict[str, any]], history: List[Dict[str, str]], language: str) -> str:
    """Helper function to construct the final prompt for the answer generation."""
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
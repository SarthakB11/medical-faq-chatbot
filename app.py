import streamlit as st
import os

from src.retriever import retrieve_context
from src.answer_generator import generate_answer
from src.config import DB_PATH, VECTOR_STORE_EXISTS

# --- Streamlit App ---
st.set_page_config(page_title="Medical FAQ Chatbot", page_icon="⚕️")
st.title("⚕️ RAG-based Medical FAQ Chatbot")
st.markdown("Ask a medical question and get an answer from our knowledge base.")

if not VECTOR_STORE_EXISTS:
    st.error(
        "The vector store database was not found. "
        f"Please run `build_vector_store.py` to create it at: {DB_PATH}"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your medical question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        # Get the last few messages for context, excluding the current prompt
        history = st.session_state.messages[:-1]
        
        retrieved_docs = retrieve_context(prompt)
        if not retrieved_docs:
            response = "I could not find any relevant information in the knowledge base to answer your question."
        else:
            response = generate_answer(prompt, retrieved_docs, history=history, language="English")

    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
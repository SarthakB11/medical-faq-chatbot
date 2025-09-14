import streamlit as st
from src.retriever import retrieve_context
from src.answer_generator import generate_answer
import os

# --- Configuration ---
DB_PATH = 'chroma_db'
COLLECTION_NAME = 'medical_faqs'
# Check if the vector store has been built
VECTOR_STORE_EXISTS = os.path.exists(DB_PATH)

# --- Streamlit App ---

st.set_page_config(page_title="Medical FAQ Chatbot", page_icon="⚕️")

st.title("⚕️ RAG-based Medical FAQ Chatbot")
st.markdown("Ask a medical question and get an answer from our knowledge base.")

# --- Initialization Check ---
if not VECTOR_STORE_EXISTS:
    st.error(
        "The vector store has not been created yet. "
        "Please run the `build_vector_store.py` script first to initialize the database."
    )
    st.stop()

# --- Chat Interface ---

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your medical question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # --- RAG Pipeline ---
    with st.spinner("Thinking..."):
        # 1. Retrieve context
        st.write("Retrieving relevant context from the knowledge base...")
        retrieved_docs = retrieve_context(prompt, collection_name=COLLECTION_NAME, db_path=DB_PATH)

        if not retrieved_docs:
            response = "I could not find any relevant information in the knowledge base to answer your question."
        else:
            # 2. Generate answer
            st.write("Generating an answer based on the context...")
            # For simplicity, we'll assume the language is English.
            # A more advanced version could detect the language of the prompt.
            response = generate_answer(prompt, retrieved_docs, language="English")

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

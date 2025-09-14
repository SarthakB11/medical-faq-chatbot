import streamlit as st
import os
import json
import datetime
from src.retriever import retrieve_context
from src.answer_generator import generate_answer
from src.config import DB_PATH, VECTOR_STORE_EXISTS

# --- Feedback Logging ---
FEEDBACK_LOG_FILE = "feedback.log"

def log_feedback(question, answer, feedback):
    """Logs user feedback to a file."""
    with open(FEEDBACK_LOG_FILE, "a") as f:
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "feedback": feedback
        }
        f.write(json.dumps(log_entry) + "\n")

# --- Streamlit App ---
st.set_page_config(page_title="Medical FAQ Chatbot", page_icon="⚕️")
st.title("⚕️ RAG-based Medical FAQ Chatbot")
st.markdown("Ask a medical question and get an answer from our knowledge base.")

if not VECTOR_STORE_EXISTS:
    st.error(f"Vector store not found. Please run `build_vector_store.py`.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Add feedback buttons for assistant messages
        if message["role"] == "assistant" and i > 0:
            # Get the question that prompted this answer
            question = st.session_state.messages[i-1]["content"]
            
            col1, col2, _ = st.columns([1, 1, 10])
            with col1:
                if st.button("👍", key=f"up_{i}"):
                    log_feedback(question, message["content"], "positive")
                    st.success("Thanks for your feedback!")
            with col2:
                if st.button("👎", key=f"down_{i}"):
                    log_feedback(question, message["content"], "negative")
                    st.error("Thanks for your feedback!")

# Handle new user input
if prompt := st.chat_input("What is your medical question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        history = [msg for msg in st.session_state.messages if msg["role"] != "user"][-4:] # Get last 2 pairs
        retrieved_docs = retrieve_context(prompt)
        
        if not retrieved_docs:
            response = "I could not find any relevant information to answer your question."
        else:
            response = generate_answer(prompt, retrieved_docs, history=history)

    with st.chat_message("assistant"):
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Rerun to display the new message with feedback buttons
    st.rerun()

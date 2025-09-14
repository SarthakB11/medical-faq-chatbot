import streamlit as st
import os
import json
import datetime
from src.retriever import retrieve_context
from src.answer_generator import generate_answer_stream, rewrite_query
from src.config import DB_PATH

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

if not os.path.exists(DB_PATH):
    st.error(f"Vector store not found. Please run `build_vector_store.py`.")
    st.stop()


if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and i > 0:
            question = st.session_state.messages[i-1]["content"]
            col1, col2, _ = st.columns([1, 1, 10])
            if col1.button("👍", key=f"up_{i}"):
                log_feedback(question, message["content"], "positive")
                st.success("Thanks for your feedback!")
            if col2.button("👎", key=f"down_{i}"):
                log_feedback(question, message["content"], "negative")
                st.error("Thanks for your feedback!")

# Handle new user input
if prompt := st.chat_input("What is your medical question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Rewriting query and searching..."):
            history = st.session_state.messages[:-1][-4:]
            
            # 1. Rewrite the query
            rewritten = rewrite_query(prompt, history)
            st.info(f"Searching for: _{rewritten}_") # Show the user the rewritten query
            
            # 2. Retrieve context with the rewritten query
            retrieved_docs = retrieve_context(rewritten, threshold=0.0)
            
            if not retrieved_docs:
                response = "I could not find any relevant information to answer your question."
                st.markdown(response)
            else:
                # 3. Generate the answer with the original query and the retrieved context
                response = st.write_stream(generate_answer_stream(prompt, retrieved_docs))
    
    st.session_state.messages.append({"role": "assistant", "content": response})

# src/config.py

import os

# --- File Paths ---
# Use absolute paths to prevent issues with the working directory
# __file__ is the path to the current script (config.py)
# os.path.dirname(__file__) is the 'src' directory
# os.path.abspath(...) gets the absolute path
# The project root is one level up from the 'src' directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

DB_PATH = os.path.join(PROJECT_ROOT, 'chroma_db')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'medical_faqs.csv')

# --- ChromaDB Configuration ---
COLLECTION_NAME = "medical_faqs"

# --- Model Configuration ---
EMBEDDING_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'
LLM_MODEL_NAME = 'gemini-2.0-flash'

# --- Retriever Configuration ---
CONTEXT_RETRIEVAL_N_RESULTS = 3
CONTEXT_RETRIEVAL_THRESHOLD = 0.5 # Using a threshold can help filter out irrelevant results

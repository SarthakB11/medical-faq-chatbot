# RAG-based Medical FAQ Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer medical questions using a private knowledge base. It is built with a modular architecture, supports conversational follow-up questions, and cites its sources to ensure accuracy and trust.

## Features

- **Advanced RAG Pipeline**: Retrieves relevant context from a knowledge base before generating an answer.
- **Source Citation**: Cites the specific source document(s) used to generate an answer, allowing for verification.
- **Conversation Memory**: Remembers the last few turns of the conversation to understand follow-up questions.
- **Streaming Responses**: The web interface streams answers token-by-token for a real-time chat experience.
- **Multilingual Support**: Can understand and respond to queries in multiple languages.
- **User Feedback**: The web interface includes a thumbs-up/thumbs-down mechanism to log feedback on answer quality.
- **Modular and Extensible**: Built with a clean, abstracted architecture that makes it easy to add new features or swap components like the language model.

## How It Works

1.  **Data Preprocessing**: Medical FAQs are loaded from a CSV file. Each FAQ is assigned a unique `source_id`.
2.  **Vector Store**: The FAQs are converted into vector embeddings and stored in a ChromaDB database with their `source_id` as metadata.
3.  **Retrieval**: When a user asks a question, the query is embedded, and the most relevant text chunks (including their source metadata) are retrieved from the database.
4.  **Generation**: The user's query, the conversation history, and the retrieved context (with source IDs) are passed to the Gemini 2.0 Flash language model. The model is instructed to answer the question and cite the sources it used.

## Prerequisites

- Python 3.8+
- A Google Gemini API key

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/SarthakB11/medical-faq-chatbot.git
    cd medical-faq-chatbot
    ```

2.  **Create a virtual environment and install dependencies:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Set up your Gemini API key:**
    Create a `.env` file by copying the example: `cp .env.example .env`. Then, add your API key to the `.env` file.

4.  **Build the vector store:**
    Run the following script to process the dataset in the `data/` directory and create the vector database:
    ```bash
    venv/bin/python3 src/build_vector_store.py
    ```

## Usage

### Streamlit Web App (Recommended)

The web app provides the full experience, including streaming responses and feedback buttons.
```bash
venv/bin/streamlit run app.py
```

### Command-Line Interface

The CLI supports interactive, multi-turn conversations with automatic language detection.
```bash
venv/bin/python3 cli.py
```

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── cli.py                  # Command-line interface
├── requirements.txt        # Project dependencies
├── .env.example
├── data/
│   └── medical_faqs.csv    # Knowledge base
├── src/
│   ├── config.py           # Centralized configuration
│   ├── data_loader.py
│   ├── build_vector_store.py
│   ├── retriever.py
│   ├── llm.py              # Language model abstraction
│   └── answer_generator.py
└── tests/
    └── ...                 # Unit tests for each module
```
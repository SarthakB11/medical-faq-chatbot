# RAG-based Medical FAQ Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot designed to answer medical questions using a knowledge base of medical FAQs. It leverages a vector database and OpenAI's language models to provide accurate and natural-sounding answers.

## Features

- **RAG Pipeline**: Retrieves relevant context from a knowledge base before generating an answer.
- **Multilingual Support**: Can understand and respond to queries in multiple languages.
- **Simple User Interfaces**: Interact with the chatbot via a Streamlit web app or a command-line interface.
- **Modular Design**: The codebase is organized into modules for data loading, vector store management, retrieval, and answer generation.

## How It Works

1.  **Data Preprocessing**: Medical FAQs are loaded from a CSV file, chunked into smaller pieces, and converted into vector embeddings using a multilingual Sentence Transformer model.
2.  **Vector Store**: The embeddings are stored in a ChromaDB vector database for efficient similarity search.
3.  **Retrieval**: When a user asks a question, the query is embedded, and the most relevant text chunks are retrieved from the vector store.
4.  **Generation**: The user's query and the retrieved context are passed to an OpenAI language model (GPT-3.5-turbo) to generate a final answer.

## Prerequisites

- Python 3.8+
- An OpenAI API key

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

3.  **Set up your OpenAI API key:**
    Create a `.env` file in the root directory by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Then, open the `.env` file and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your-openai-api-key-here"
    ```

4.  **Prepare the dataset:**
    Place your medical FAQ dataset in the `data` directory. A sample file named `dummy_medical_data.csv` is provided.

5.  **Build the vector store:**
    Run the following script to process the dataset and create the vector database:
    ```bash
    venv/bin/python3 src/build_vector_store.py
    ```

## Usage

### Streamlit Web App

To launch the web interface, run:
```bash
venv/bin/streamlit run app.py
```
Open your browser and navigate to the URL provided (usually `http://localhost:8501`).

### Command-Line Interface

To use the CLI, run:
```bash
venv/bin/python3 cli.py "Your medical question here"
```
You can also specify the language for the answer:
```bash
venv/bin/python3 cli.py "qué es la fiebre" --lang Spanish
```

## Project Structure

```
.
├── app.py                  # Streamlit web application
├── cli.py                  # Command-line interface
├── requirements.txt        # Project dependencies
├── .env.example            # Example environment file
├── data/                   # Directory for datasets
│   └── dummy_medical_data.csv
├── src/                    # Source code
│   ├── data_loader.py
│   ├── build_vector_store.py
│   ├── retriever.py
│   └── answer_generator.py
└── tests/                  # Unit tests
    ├── test_data_loader.py
    ├── test_vector_store.py
    ├── test_retriever.py
    └── test_answer_generator.py
```

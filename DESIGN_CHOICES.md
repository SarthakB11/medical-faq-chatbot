# Design Choices for the RAG-based Medical FAQ Chatbot

This document outlines the key design and technology choices made during the development of this project.

## 1. Core Architecture: Retrieval-Augmented Generation (RAG)

The RAG architecture was chosen because it is highly effective for building chatbots that need to answer questions based on a specific knowledge base.

-   **Why RAG?**: Instead of relying solely on the general knowledge of a large language model (LLM), RAG first retrieves relevant information from a trusted source. This significantly reduces the risk of hallucinations (the model making up facts) and ensures that the answers are grounded in the provided medical data.

## 2. Technology Stack

### a. Vector Database: ChromaDB

-   **Choice**: ChromaDB was selected as the vector database.
-   **Reasoning**:
    -   **Simplicity**: It is lightweight and can be run locally without requiring a separate server, making it ideal for a self-contained project.
    -   **Ease of Use**: The API is straightforward for creating collections, adding embeddings, and performing similarity searches.
    -   **Performance**: For the scale of this project (~100-1000 FAQs), ChromaDB's performance is more than sufficient.

### b. Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`

-   **Choice**: A multilingual model from the Sentence Transformers library.
-   **Reasoning**:
    -   **Multilingual Support**: This was a key requirement. This model is capable of understanding and comparing text across many different languages, allowing the chatbot to be queried in languages other than English.
    -   **Performance vs. Size**: It offers a great balance between embedding quality and model size, making it efficient to run locally. While larger models might provide slightly better results, this model is a practical choice for a project with constraints on free tools.

### c. Language Model (LLM): OpenAI's `gpt-3.5-turbo`

-   **Choice**: OpenAI's GPT-3.5-turbo.
-   **Reasoning**:
    -   **High-Quality Generation**: It is excellent at understanding context and generating fluent, natural-sounding text.
    -   **Instruction Following**: It reliably follows the instructions in the system prompt, such as answering only from the provided context and responding in a specific language.
    -   **Accessibility**: It is widely available and offers a free tier, aligning with the project's constraints.

### d. User Interfaces: Streamlit and CLI

-   **Choice**: Both a web app (Streamlit) and a command-line interface (CLI) were created.
-   **Reasoning**:
    -   **Streamlit**: This library was chosen for the web app because it allows for the rapid development of interactive and visually appealing interfaces with minimal code. It's perfect for demonstrating the chatbot's functionality in an accessible way.
    -   **CLI**: A command-line interface provides a simple, lightweight way to interact with the chatbot, which is useful for testing, scripting, and for users who prefer a terminal-based workflow.

## 3. Modularity

The project is broken down into distinct Python modules (`data_loader`, `build_vector_store`, `retriever`, `answer_generator`).

-   **Reasoning**:
    -   **Clarity and Maintainability**: This separation of concerns makes the code easier to read, understand, and maintain.
    -   **Testability**: Each component can be tested independently, which was demonstrated in the unit tests for each module.
    -   **Reusability**: The modular design would make it easier to swap out components in the future (e.g., changing the vector database or the LLM).

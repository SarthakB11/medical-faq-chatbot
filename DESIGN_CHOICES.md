# Design Choices for the RAG-based Medical FAQ Chatbot

This document outlines the key design and technology choices made during the development of this project.

## 1. Core Architecture: Retrieval-Augmented Generation (RAG)

The RAG architecture was chosen because it is highly effective for building chatbots that need to answer questions based on a specific knowledge base.

-   **Why RAG?**: Instead of relying solely on the general knowledge of a large language model (LLM), RAG first retrieves relevant information from a trusted source. This significantly reduces the risk of hallucinations (the model making up facts) and ensures that the answers are grounded in the provided medical data.

## 2. Architectural Patterns

### a. Centralized Configuration

-   **Choice**: All key configuration variables (file paths, model names, retriever settings) are centralized in the `src/config.py` file.
-   **Reasoning**: This pattern makes the application easier to manage and modify. Instead of searching for hardcoded values across multiple files, a developer can change a setting in one place, and it will be applied globally.

### b. Language Model Abstraction

-   **Choice**: The language model is abstracted into a `LanguageModel` base class in `src/llm.py`, with a concrete `GeminiModel` implementation.
-   **Reasoning**: This decouples the core application logic from the specific LLM provider. It makes the system more flexible and extensible. Adding a new provider (like OpenAI or a local model) in the future would only require creating a new class that inherits from `LanguageModel`, without changing the `answer_generator` or UI code.

## 3. Technology Stack

### a. Vector Database: ChromaDB

-   **Choice**: ChromaDB was selected as the vector database.
-   **Reasoning**: Simplicity, ease of use for local development, and sufficient performance for the project's scale.

### b. Embedding Model: `paraphrase-multilingual-MiniLM-L12-v2`

-   **Choice**: A multilingual model from the Sentence Transformers library.
-   **Reasoning**: Strong balance of performance and size, and crucial for the multilingual support requirement.

### c. Language Model (LLM): Google's `gemini-2.0-flash`

-   **Choice**: Google's Gemini 2.0 Flash model.
-   **Reasoning**: High-quality generation, excellent instruction-following capabilities, and accessibility via a free tier.

### d. User Interfaces: Streamlit and CLI

-   **Choice**: Both a web app (Streamlit) and a command-line interface (CLI) were created.
-   **Reasoning**: Streamlit for rapid, interactive UI development, and a CLI for testing and scripting.

## 4. Key Features

### a. Source Citation

-   **Implementation**: Each document is tagged with a `source_id` during data loading. This ID is stored as metadata in the vector database, retrieved with the context, and included in the prompt to the LLM with an instruction to cite it.
-   **Reasoning**: This is a critical feature for any RAG system, especially in a medical context, as it builds user trust and allows for fact-checking.

### b. Conversation Memory

-   **Implementation**: The Streamlit app and CLI both maintain a history of the conversation. This history is passed to the LLM in the prompt, providing context for follow-up questions.
-   **Reasoning**: This transforms the chatbot from a simple Q&A tool into a more natural conversational agent.

### c. Streaming Responses

-   **Implementation**: The `GeminiModel` was designed with a `generate_stream` method. The Streamlit app uses `st.write_stream` to consume this stream, displaying the answer token by token.
-   **Reasoning**: This significantly improves the perceived performance and user experience of the web app, as the user sees an immediate response.
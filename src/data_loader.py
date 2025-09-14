import pandas as pd
import os
from typing import List, Dict

def load_data(file_path: str) -> List[Dict[str, str]]:
    """
    Loads medical FAQs from a CSV file.

    Args:
        file_path: The path to the CSV file.

    Returns:
        A list of dictionaries, where each dictionary represents an FAQ
        with 'question' and 'answer' keys.
        
    Raises:
        FileNotFoundError: If the file is not found at the specified path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file was not found at: {file_path}")

    df = pd.read_csv(file_path)
    
    # Simple preprocessing: drop rows with missing values
    df.dropna(subset=['Question', 'Answer'], inplace=True)

    # Add a source identifier based on the row number
    df['source_id'] = df.index.map(lambda x: f"FAQ-{x+1}")

    # Combine question and answer for a single text block per FAQ
    df['text'] = df['Question'] + " " + df['Answer']

    return df.to_dict('records')

def chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Splits a long text into smaller chunks.
    
    Args:
        text: The text to be chunked.
        chunk_size: The maximum size of each chunk.
        overlap: The number of characters to overlap between chunks.

    Returns:
        A list of text chunks.
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

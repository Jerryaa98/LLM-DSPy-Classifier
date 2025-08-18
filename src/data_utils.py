"""
Data utilities for the DSPy project.
"""
import os
import pandas as pd
from typing import List, Dict, Any

# Simple example dataset
SAMPLE_DATA = [
    {
        "context": "The United States is a country primarily located in North America. It consists of 50 states, a federal district, five major unincorporated territories, 326 Indian reservations, and nine minor outlying islands.",
        "question": "How many states are in the United States?",
        "answer": "50 states"
    },
    {
        "context": "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically typed and garbage-collected.",
        "question": "What programming language emphasizes code readability?",
        "answer": "Python"
    },
    {
        "context": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. It is named after the engineer Gustave Eiffel, whose company designed and built the tower.",
        "question": "Who designed the Eiffel Tower?",
        "answer": "Gustave Eiffel"
    }
]

def load_sample_data() -> List[Dict[str, str]]:
    """
    Load a sample dataset for question answering.
    """
    return SAMPLE_DATA

def save_dataset(data: List[Dict[str, Any]], filename: str = "sample_dataset.csv") -> str:
    """
    Save a dataset to a CSV file in the data directory.
    
    Args:
        data: List of dictionaries containing the dataset
        filename: Name of the file to save
        
    Returns:
        Path to the saved file
    """
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    filepath = os.path.join(data_dir, filename)
    df = pd.DataFrame(data)
    df.to_csv(filepath, index=False)
    
    return filepath

def load_dataset(filepath: str) -> List[Dict[str, Any]]:
    """
    Load a dataset from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of dictionaries containing the dataset
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    df = pd.DataFrame(pd.read_csv(filepath))
    return df.to_dict('records')

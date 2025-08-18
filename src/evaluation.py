"""
Evaluation utilities for the DSPy project.
"""
from typing import List, Dict, Any

def evaluate_qa(predictions: List[Dict[str, Any]], references: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Evaluate question answering predictions against reference answers.
    
    Args:
        predictions: List of dictionaries containing predictions
        references: List of dictionaries containing reference answers
        
    Returns:
        Dictionary containing evaluation metrics
    """
    if len(predictions) != len(references):
        raise ValueError("Number of predictions and references must match")
    
    correct = 0
    total = len(predictions)
    
    for pred, ref in zip(predictions, references):
        # Simple exact match evaluation
        pred_answer = pred.get("answer", "").lower().strip()
        ref_answer = ref.get("answer", "").lower().strip()
        
        if pred_answer in ref_answer or ref_answer in pred_answer:
            correct += 1
    
    accuracy = correct / total if total > 0 else 0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }

def print_evaluation_results(results: Dict[str, float]) -> None:
    """
    Print evaluation results in a readable format.
    
    Args:
        results: Dictionary containing evaluation metrics
    """
    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {results['accuracy']:.2f} ({results['correct']}/{results['total']})")
    print("="*25)

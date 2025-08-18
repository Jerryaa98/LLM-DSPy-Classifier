"""
Utility script for analyzing the results of the Q&A experiments.
"""
import os
import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_accuracy_comparison(results: Dict[str, float], output_path: str = None):
    """Plot a comparison of accuracy across different Q&A systems."""
    methods = list(results.keys())
    accuracies = [results[method] for method in methods]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, accuracies, color=['blue', 'orange', 'green'])
    
    # Add data labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom')
    
    plt.title('Accuracy Comparison Across Q&A Systems')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1.1)  # Accuracy from 0 to 1
    plt.grid(axis='y', alpha=0.3)
    
    if output_path:
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def analyze_error_patterns(predictions: List[Dict[str, Any]], dataset: pd.DataFrame):
    """Analyze error patterns in predictions, especially related to spurious correlations."""
    # Count errors by different conditions
    errors_with_ceremony = 0
    errors_without_ceremony = 0
    total_with_ceremony = 0
    total_without_ceremony = 0
    
    for pred in predictions:
        article_id = pred["article_id"]
        article = dataset[dataset["article_id"] == article_id].iloc[0]
        
        is_error = pred["is_correct"] == False
        has_ceremony = article["has_ceremony"]
        
        if has_ceremony:
            total_with_ceremony += 1
            if is_error:
                errors_with_ceremony += 1
        else:
            total_without_ceremony += 1
            if is_error:
                errors_without_ceremony += 1
    
    # Calculate error rates
    error_rate_with_ceremony = errors_with_ceremony / total_with_ceremony if total_with_ceremony > 0 else 0
    error_rate_without_ceremony = errors_without_ceremony / total_without_ceremony if total_without_ceremony > 0 else 0
    
    print(f"Error rate with ceremony mention: {error_rate_with_ceremony:.2f}")
    print(f"Error rate without ceremony mention: {error_rate_without_ceremony:.2f}")
    
    # Check if the model is being influenced by the spurious correlation
    if error_rate_with_ceremony > error_rate_without_ceremony:
        print("The model appears to be influenced by the spurious correlation (ceremony mentions).")
    else:
        print("The model does not seem to be strongly influenced by the spurious correlation.")
    
    return {
        "error_rate_with_ceremony": error_rate_with_ceremony,
        "error_rate_without_ceremony": error_rate_without_ceremony,
        "total_with_ceremony": total_with_ceremony,
        "total_without_ceremony": total_without_ceremony
    }

def save_results(results: Dict[str, Any], output_path: str):
    """Save experiment results to a JSON file."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_path}")

def main():
    """Main function to run analysis on experiment results."""
    # Example usage (this would normally load actual results)
    example_results = {
        "Basic QA": 0.65,
        "Unoptimized QA": 0.72,
        "Optimized QA": 0.85
    }
    
    # Plot results
    plot_accuracy_comparison(example_results, "accuracy_comparison.png")
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()

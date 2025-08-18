"""
Main script for running the DSPy Q&A pipeline.
"""
import os
import sys
import random
import pandas as pd
import dspy
from typing import List, Dict, Any, Tuple

# Add the project directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
from src.config import (
    OPENROUTER_API_KEY, 
    DEFAULT_MODEL, 
    EVALUATION_MODEL, 
    DATASET_PATH, 
    NUM_EXAMPLES,
    NUM_OPTIMIZATION_STEPS,
    OPENROUTER_BASE_URL,
    HTTP_REFERER
)
from src.qa_modules import BasicQA, OptimizableQA, QAEvaluator
from src.openrouter_client import create_openrouter_client

def load_dataset(dataset_path: str) -> pd.DataFrame:
    """Load the dataset from CSV."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Run data_generator.py first.")
    
    return pd.DataFrame(pd.read_csv(dataset_path))

def generate_qa_pairs(df: pd.DataFrame, num_examples: int = 50) -> List[Dict[str, Any]]:
    """Generate question-answer pairs from the dataset."""
    qa_pairs = []
    
    # Sample random articles
    sampled_articles = df.sample(min(num_examples, len(df)))
    
    for _, article in sampled_articles.iterrows():
        # Generate different types of questions
        
        # Basic factual question about the winner
        qa_pairs.append({
            "context": article["content"],
            "question": f"Who won the match described in this article?",
            "reference_answer": article["winner"]
        })
        
        # Question that might be influenced by spurious correlation
        qa_pairs.append({
            "context": article["content"],
            "question": f"Did the Blue Lions win this match?",
            "reference_answer": "Yes" if article["winner"] == "Blue Lions" else "No"
        })
        
        # Question about the score
        if "score" in article["title"]:
            qa_pairs.append({
                "context": article["content"],
                "question": "What was the score of the match?",
                "reference_answer": article["title"].split("Defeats")[1].split("in")[0].strip()
            })
    
    return qa_pairs

def configure_dspy(api_key: str, model_name: str, base_url: str, http_referer: str) -> None:
    """Configure DSPy with the OpenRouter client."""
    # Set up OpenRouter integration
    openrouter_lm = create_openrouter_client(
        api_key=api_key,
        model=model_name,
        base_url=base_url,
        http_referer=http_referer
    )
    dspy.settings.configure(lm=openrouter_lm)
    
    print(f"DSPy configured with OpenRouter model: {model_name}")

def evaluate_qa_system(qa_system, test_examples: List[Dict[str, Any]], evaluator) -> Dict[str, float]:
    """Evaluate a Q&A system on test examples."""
    scores = []
    correct_answers = 0
    
    for example in test_examples:
        # Get prediction
        prediction = qa_system(context=example["context"], question=example["question"])
        # print(prediction)
        # input()
        
        # Evaluate
        eval_result = evaluator(
            context=example["context"],
            question=example["question"],
            reasoning=prediction["reasoning"],
            answer=prediction["answer"],
            reference_answer=example["reference_answer"]
        )
        
        # Check if answer is correct (simple string match)
        is_correct = example["reference_answer"].lower() in prediction["answer"].lower()
        if is_correct:
            correct_answers += 1
        
        scores.append(float(eval_result["score"]))
    
    # Calculate metrics
    avg_score = sum(scores) / len(scores) if scores else 0
    accuracy = correct_answers / len(test_examples) if test_examples else 0
    
    return {
        "average_score": avg_score,
        "accuracy": accuracy
    }

def train_test_split(examples: List[Dict[str, Any]], test_size: float = 0.2) -> Tuple[List, List]:
    """Split examples into training and test sets."""
    random.shuffle(examples)
    split_idx = int(len(examples) * (1 - test_size))
    return examples[:split_idx], examples[split_idx:]

def main():
    """Main function to run the DSPy Q&A pipeline."""
    # Check if API key is set
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY not found. Set it in .env file or directly in config.py")
    
    # Configure DSPy with OpenRouter
    configure_dspy(
        api_key=OPENROUTER_API_KEY, 
        model_name=DEFAULT_MODEL,
        base_url=OPENROUTER_BASE_URL,
        http_referer=HTTP_REFERER
    )
    
    # Load dataset
    print("Loading dataset...")
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} articles")
    
    # Generate QA pairs
    print("Generating question-answer pairs...")
    qa_pairs = generate_qa_pairs(df, NUM_EXAMPLES)
    print(f"Generated {len(qa_pairs)} QA pairs")
    
    # Split into train/test
    train_examples, test_examples = train_test_split(qa_pairs)
    print(f"Split into {len(train_examples)} training and {len(test_examples)} testing examples")
    
    # Create evaluator
    evaluator = QAEvaluator()
    
    # Experiment 1: Basic QA
    print("\n--- Experiment 1: Basic QA ---")
    basic_qa = BasicQA()
    basic_qa_metrics = evaluate_qa_system(basic_qa, test_examples, evaluator)
    print(f"Basic QA Metrics: {basic_qa_metrics}")
    
    # Experiment 2: Optimizable QA without optimization
    print("\n--- Experiment 2: Optimizable QA without optimization ---")
    optimizable_qa = OptimizableQA()
    optimizable_qa_metrics = evaluate_qa_system(optimizable_qa, test_examples, evaluator)
    print(f"Optimizable QA (No Opt) Metrics: {optimizable_qa_metrics}")
    
    # Experiment 3: Optimizable QA with optimization
    print("\n--- Experiment 3: Optimizable QA with optimization ---")
    
    # Create a teleprompter for optimization
    teleprompter = dspy.Teleprompter(OptimizableQA)
    
    # Define a simple metric for optimization
    def qa_metric(example, pred):
        reference = example["reference_answer"].lower()
        prediction = pred["answer"].lower()
        return float(reference in prediction)
    
    # Optimize using the teleprompter
    print(f"Optimizing QA system over {NUM_OPTIMIZATION_STEPS} steps...")
    optimized_qa = teleprompter.compile(
        trainset=train_examples,
        metric=qa_metric,
        num_threads=1,
        max_bootstrapped_demos=3,
        num_optimization_steps=NUM_OPTIMIZATION_STEPS
    )
    
    # Evaluate optimized system
    optimized_qa_metrics = evaluate_qa_system(optimized_qa, test_examples, evaluator)
    print(f"Optimized QA Metrics: {optimized_qa_metrics}")
    
    # Print comparison
    print("\n--- Results Comparison ---")
    print(f"Basic QA Accuracy: {basic_qa_metrics['accuracy']:.2f}")
    print(f"Unoptimized QA Accuracy: {optimizable_qa_metrics['accuracy']:.2f}")
    print(f"Optimized QA Accuracy: {optimized_qa_metrics['accuracy']:.2f}")

if __name__ == "__main__":
    main()

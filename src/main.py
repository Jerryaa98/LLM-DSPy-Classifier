"""
Main script for running the DSPy Q&A pipeline with OpenRouter via LiteLLM.
"""
import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

# Import project modules
from dspy_integration import create_qa_system
from data_utils import load_sample_data, save_dataset
from evaluation import evaluate_qa, print_evaluation_results

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run DSPy Q&A pipeline with OpenRouter")
    parser.add_argument(
        "--model", 
        type=str, 
        default="openrouter/mistralai/mistral-small-3.1-24b-instruct:free",
        help="OpenRouter model to use"
    )
    parser.add_argument(
        "--save-data", 
        action="store_true", 
        help="Save sample data to CSV"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Print debug information"
    )
    return parser.parse_args()

def run_qa_pipeline(model_name, debug=False):
    """Run the Q&A pipeline using DSPy and OpenRouter."""
    # Create the QA system
    print(f"Creating QA system with model: {model_name}")
    qa_system = create_qa_system(model_name)
    
    # Load sample data
    print("Loading sample data...")
    sample_data = load_sample_data()
    
    # Run predictions
    print("Running predictions...")
    predictions = []
    for example in sample_data:
        if debug:
            print(f"\nQuestion: {example['question']}")
            print(f"Context: {example['context'][:100]}...")
        
        prediction = qa_system(context=example['context'], question=example['question'])
        predictions.append(prediction)
        
        if debug:
            print(f"Predicted Answer: {prediction['answer']}")
            print(f"Reference Answer: {example['answer']}")
    
    # Evaluate predictions
    print("Evaluating predictions...")
    eval_results = evaluate_qa(predictions, sample_data)
    print_evaluation_results(eval_results)
    
    return predictions, eval_results

def main():
    """Main function to run the DSPy Q&A pipeline."""
    args = parse_args()
    
    # Check if API key is set
    if not os.environ.get('OPENROUTER_API_KEY'):
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Save sample data if requested
    if args.save_data:
        sample_data = load_sample_data()
        filepath = save_dataset(sample_data)
        print(f"Sample data saved to {filepath}")
    
    # Run the pipeline
    run_qa_pipeline(args.model, args.debug)

if __name__ == "__main__":
    main()

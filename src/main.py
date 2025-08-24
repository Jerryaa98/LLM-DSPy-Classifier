"""
Script to load a CSV of math questions, answer them with OpenRouter, and compare to the reference answer.

This script demonstrates how to:
1. Load a dataset of math addition questions from a CSV file.
2. Use OpenRouter (via LiteLLM) to answer each question.
3. Compare the model's answer to the reference answer and print the results.
4. Calculate and print the overall accuracy.
"""

import os
import pandas as pd
from litellm import completion
from dotenv import load_dotenv
import argparse

# Load environment variables from .env file
load_dotenv()
# Set the OpenRouter API key for LiteLLM
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

def ask_openrouter(prompt, model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
    """
    Send a prompt to the OpenRouter model using LiteLLM and return the model's response as a string.
    Args:
        prompt (str): The question or instruction to send to the model.
        model (str): The OpenRouter model to use.
    Returns:
        str: The model's answer as a string.
    """
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    # Try to extract the answer as a string
    try:
        return response.choices[0].message.content.strip()
    except Exception:
        return str(response)

def main_math():
    """
    Main function to load math questions, answer them with OpenRouter, and compare to reference answers.
    """
    # Path to the provided math questions CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "math_addition_questions.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Load questions from CSV into a DataFrame
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} math questions.")

    correct = 0  # Counter for correct answers
    total = 0    # Counter for total questions

    # Iterate over each question in the dataset
    for idx, row in df.iterrows():
        question = row['question']
        reference = str(row['answer']).strip()
        # Add a system prompt to encourage the model to answer with only the number
        model_role = (
            "You are a math student who is asked some basic addition questions. "
            "Answer me only the number without any additional context. "
        )
        # Send the prompt to the model
        model_name = "openrouter/mistralai/mistral-small-3.1-24b-instruct:free"
        answer = ask_openrouter(prompt = model_role + question,
                                model=model_name) # Uses Mistral by default
        # Compare the model's answer to the reference answer
        is_correct = (str(answer) == str(reference))
        # Print the question, model answer, reference, and correctness
        print(f"Q: {question}\nOpenRouter Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1

        if total >= 10: # Limit to 10 questions for free models bottleneck
            break

    # Print the overall accuracy
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")
    
def main_md():
    """
    Main function to load Markdown content, answer with OpenRouter, and compare to reference answers.
    """
    # Path to the provided Markdown content CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "md_content_classification.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Load questions from CSV into a DataFrame
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} Markdown content samples.")

    correct = 0  # Counter for correct answers
    total = 0    # Counter for total questions

    # Iterate over each question in the dataset
    for idx, row in df.iterrows():
        question = row['question'] + row['context']
        reference = str(row['answer']).strip()
        # Add a system prompt to encourage the model to answer with only yes/no
        model_role = (
            "You are a website classification model. "
            "Classify the following Markdown content as a funding opportunity or not using only yes or no as an answer "
            "without any other additions even a point: "
        )
        # Send the prompt to the model
        model_name = "openrouter/mistralai/mistral-small-3.1-24b-instruct:free"
        answer = ask_openrouter(prompt=model_role + question,
                                model=model_name)
        # Compare the model's answer to the reference answer
        is_correct = (str(answer).strip().lower() == str(reference).strip().lower())
        # Print the question, model answer, reference, and correctness
        print(f"Q: {question}\nOpenRouter Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1

    # Print the overall accuracy
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")

def main_html():
    """
    Main function to load HTML content, answer with OpenRouter, and compare to reference answers.
    """
    # Path to the provided HTML content CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "html_content_classification.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Load questions from CSV into a DataFrame
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} HTML content samples.")

    correct = 0  # Counter for correct answers
    total = 0    # Counter for total questions

    # Iterate over each question in the dataset
    for idx, row in df.iterrows():
        question = row['question'] + row['context']
        reference = str(row['answer']).strip()
        # Add a system prompt to encourage the model to answer with only yes/no
        model_role = (
            "You are a website classification model. "
            "Classify the following HTML content as a funding opportunity or not using only yes or no as an answer "
            "without any other additions even a point: "
        )
        # Send the prompt to the model
        model_name = "openrouter/mistralai/mistral-small-3.1-24b-instruct:free"
        answer = ask_openrouter(prompt=model_role + question,
                                model=model_name)
        # Compare the model's answer to the reference answer
        is_correct = (str(answer).strip().lower() == str(reference).strip().lower())
        # Print the question, model answer, reference, and correctness
        print(f"Q: {question}\nOpenRouter Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1

    # Print the overall accuracy
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")
    
def main_website():
    """
    Main function to load website links, answer them with OpenRouter, and compare to reference answers.
    """
    # Path to the provided website links CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "link_content_classification.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return
    
    # Load questions from CSV into a DataFrame
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} website links.")

    correct = 0  # Counter for correct answers
    total = 0    # Counter for total questions
    
    # Iterate over each question in the dataset
    for idx, row in df.iterrows():
        question = row['question'] + row['context']
        reference = str(row['answer']).strip()
        # Add a system prompt to encourage the model to answer with only the number
        model_role = (
            "You are a website classification model. "
            "Classify the following content that was scraped off a website "
            "as a funding opportunity or not using only yes or no as an answer "
            "without any other additions even a point: "
        )
        # Send the prompt to the model
        model_name = "openrouter/mistralai/mistral-small-3.1-24b-instruct:free"
        answer = ask_openrouter(prompt = model_role + question,
                                model=model_name) # Uses Mistral by default
        # Compare the model's answer to the reference answer
        is_correct = (str(answer).strip().lower() == str(reference).strip().lower())
        # Print the question, model answer, reference, and correctness
        print(f"Q: {question}\nOpenRouter Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1

        if total >= 10: # Limit to 10 questions for free models bottleneck
            break

    # Print the overall accuracy
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenRouter evaluation scripts.")
    print('Please enter one of the arguments to run the required task:\n'
          ' - math\n'
          ' - website\n'
          ' - html\n'
          ' - md\n'
          ' (default: --task math)')
    
    parser.add_argument(
        "--task",
        choices=["math", "website", "html", "md"],
        default='math',
        help="Which main function to run: math, website, html, or md (default: math)"
    )
    args = parser.parse_args()

    if args.task == "math":
        main_math()
    elif args.task == "website":
        main_website()
    elif args.task == "html":
        main_html()
    elif args.task == "md":
        main_md()
    

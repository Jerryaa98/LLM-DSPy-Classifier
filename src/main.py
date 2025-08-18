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

def main():
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

if __name__ == "__main__":
    main()

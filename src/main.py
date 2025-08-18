"""
Script to load a CSV of math questions, answer them with OpenRouter, and compare to the reference answer.
"""
import os
import pandas as pd
from litellm import completion
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

def ask_openrouter(prompt, model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
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
    # Path to the provided math questions CSV
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    csv_path = os.path.join(data_dir, "math_addition_questions.csv")
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return

    # Load questions
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} math questions.")

    correct = 0
    total = 0
    for idx, row in df.iterrows():
        question = row['question']
        reference = str(row['answer']).strip()
        model_role = ("you are a math student which is asked some basic addition questions"
                      " answer me only the number without any additional context")
        answer = ask_openrouter(model_role + question)
        is_correct = (str(answer) == str(reference))
        print(f"Q: {question}\nOpenRouter Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1
    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")

if __name__ == "__main__":
    main()

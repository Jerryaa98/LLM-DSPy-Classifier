import os
import pandas as pd
import random

def generate_addition_questions_csv(num_questions=100, min_val=0, max_val=100, filename="math_addition_questions.csv"):
    """
    Generate a CSV file of simple math addition questions and answers.
    The file will be saved in the data/ folder.
    """
    questions = []
    for _ in range(num_questions):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        question = f"What is {a} + {b}?"
        answer = str(a + b)
        questions.append({"context": "question", "question": question, "answer": answer})
    
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    pd.DataFrame(questions).to_csv(filepath, index=False)
    print(f"Generated {num_questions} addition questions and saved to {filepath}")

if __name__ == "__main__":
    generate_addition_questions_csv()

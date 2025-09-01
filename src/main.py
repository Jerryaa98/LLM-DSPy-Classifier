import time
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

# DSPy integration
import dspy


class ClassifierModule(dspy.Module):
    def __init__(self, model_name="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
        super().__init__()
        self.model = dspy.LM(
                    model=model_name,
                    api_base="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPENROUTER_API_KEY"),
                    )
        dspy.configure(lm=self.model)
        self.question = dspy.InputField(desc = "User's description and question")
        self.answer = dspy.OutputField(desc = "1 word, Yes or No without any other additions or symbols")
        self.chain_of_thought = dspy.ChainOfThought('description_question -> one_word_answer')

    def forward(self, context, question):
        prompt = f"""
        You are a website classification model. \n
        Classify the following content as a funding opportunity or not using only yes or no as an answer\n
        without any other additions even a point: {question}\n{context}\n
        """
        answer = self.chain_of_thought(description_question=prompt).one_word_answer
        return dspy.Prediction(answer=answer)

class MathClassifierModule(dspy.Module):
    def __init__(self, model_name="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
        super().__init__()
        self.model = dspy.LM(model=model_name)

    def forward(self, context, question):
        prompt = f"{question}\n{context}\nyou are a math student,  answer only the number without any additional points or symbols."
        response = self.model(prompt)
        # Ensure response is string
        if isinstance(response, list):
            response = response[0]
        response = str(response)
        # Extract only yes/no
        answer = response.strip().split()[0].lower()
        if answer not in ["yes", "no"]:
            answer = response.strip().lower()
        return dspy.Prediction(answer=answer)

def classify_with_dspy(context, question, model_name="openrouter/mistralai/mistral-small-3.1-24b-instruct:free", math=False):
    if math:
        module = MathClassifierModule(model_name=model_name)
    else:
        module = ClassifierModule(model_name=model_name)
    pred = module(context=context, question=question)
    return pred.answer

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


    # Iterate over each question in the dataset using DSPy
    for idx, row in df.iterrows():
        context = ""  # No context for math, just the question
        question = row['question']
        reference = str(row['answer']).strip()
        # Use DSPy for classification
        answer = classify_with_dspy(context, question, math=True)
        is_correct = (str(answer).replace('.', '').replace('*', '') == str(reference).replace('.', '').replace('*', ''))
        print(f"Q: {question}\nDSPy Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
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

    # Iterate over each question in the dataset using DSPy
    for idx, row in df.iterrows():
        context = row['context']
        question = row['question']
        reference = str(row['answer']).strip()
        answer = classify_with_dspy(context, question)
        is_correct = (str(answer).strip().lower().replace('.', '').replace('*', '') == str(reference).strip().lower().replace('.', '').replace('*', ''))
        print(f"Q: {question}\nDSPy Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1

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

    # Iterate over each question in the dataset using DSPy
    for idx, row in df.iterrows():
        context = row['context']
        question = row['question']
        reference = str(row['answer']).strip()
        answer = classify_with_dspy(context, question)
        is_correct = (str(answer).strip().lower().replace('.', '').replace('*', '') == str(reference).strip().lower().replace('.', '').replace('*', ''))
        print(f"Q: {question}\nDSPy Answer: {answer}\nReference: {reference}\nCorrect: {is_correct}\n---")
        total += 1
        if is_correct:
            correct += 1

    print(f"Accuracy: {correct}/{total} = {correct/total:.2f}")
    
    
def benchmark_html_vs_md():
        """
        Compare LLM outputs for HTML and Markdown content, log differences to a CSV.
        """
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        html_csv = os.path.join(data_dir, "html_content_classification.csv")
        md_csv = os.path.join(data_dir, "md_content_classification.csv")
        log_csv = os.path.join(data_dir, "benchmark_html_vs_md.csv")

        if not os.path.exists(html_csv) or not os.path.exists(md_csv):
            print("Both html_content_classification.csv and md_content_classification.csv must exist.")
            return

        df_html = pd.read_csv(html_csv)
        df_md = pd.read_csv(md_csv)

        # Try to align by row index (assumes same order and length)
        min_len = min(len(df_html), len(df_md))
        results = []
        for idx in range(min_len):
            html_row = df_html.iloc[idx]
            md_row = df_md.iloc[idx]
            html_question = html_row['question'] + html_row['context']
            md_question = md_row['question'] + md_row['context']
            reference = str(html_row['answer']).strip()

            model_role_html = (
                "You are a website classification model. "
                "Classify the following HTML content as a funding opportunity or not using only yes or no as an answer "
                "without any other additions even a point: "
            )
            model_role_md = (
                "You are a website classification model. "
                "Classify the following Markdown content as a funding opportunity or not using only yes or no as an answer "
                "without any other additions even a point: "
            )
            model_name = "openrouter/mistralai/mistral-small-3.1-24b-instruct:free"
            html_answer = ask_openrouter(prompt=model_role_html + html_question, model=model_name).replace('.', '')
            md_answer = ask_openrouter(prompt=model_role_md + md_question, model=model_name).replace('.', '')
            html_answer_clean = str(html_answer).strip().lower()
            md_answer_clean = str(md_answer).strip().lower()
            reference_clean = reference.lower()
            match = html_answer_clean == md_answer_clean
            html_correct = html_answer_clean == reference_clean
            md_correct = md_answer_clean == reference_clean
            results.append({
                "index": idx,
                "reference": reference,
                "html_answer": html_answer,
                "md_answer": md_answer,
                "answers_match": match,
                "html_correct": html_correct,
                "md_correct": md_correct
            })
            print(f"Sample {idx}: HTML='{html_answer}' | MD='{md_answer}' | Match={match} | Ref={reference}")
            time.sleep(5)
        df_results = pd.DataFrame(results)
        df_results.to_csv(log_csv, index=False)
        html_total = len(df_results)
        html_correct = df_results['html_correct'].sum()
        md_correct = df_results['md_correct'].sum()
        html_acc = html_correct / html_total if html_total else 0
        md_acc = md_correct / html_total if html_total else 0
        print(f"\nHTML correct: {html_correct}/{html_total} = {html_acc:.2f}")
        print(f"MD correct:   {md_correct}/{html_total} = {md_acc:.2f}")
        if html_acc > md_acc:
            print("HTML classification had a higher success rate.")
        elif md_acc > html_acc:
            print("Markdown classification had a higher success rate.")
        else:
            print("Both had the same success rate.")
        # Log summary row at the end of the CSV
        summary_row = {
            'index': 'SUMMARY',
            'reference': '',
            'html_answer': '',
            'md_answer': '',
            'answers_match': '',
            'html_correct': html_correct,
            'md_correct': md_correct
        }
        df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)
        df_results.to_csv(log_csv, index=False)
        print(f"Benchmark results saved to {log_csv}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenRouter evaluation scripts.")
    print('Please enter one of the arguments to run the required task:\n'
          ' - math\n'
          ' - website\n'
          ' - html\n'
          ' - md\n'
          ' - benchmark\n'
          ' (default: --task math)')
    
    parser.add_argument(
        "--task",
        choices=["math", "website", "html", "md", "benchmark"],
        default='math',
        help="Which main function to run: math, website, html, md, or benchmark (default: math)"
    )
    args = parser.parse_args()

    if args.task == "math":
        main_math()
    elif args.task == "html":
        main_html()
    elif args.task == "md":
        main_md()
    elif args.task == "benchmark":
        benchmark_html_vs_md()
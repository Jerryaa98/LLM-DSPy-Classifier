"""
Simple script to call OpenRouter via LiteLLM completion API.
"""
import os
from litellm import completion
from dotenv import load_dotenv

load_dotenv()
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')

def ask_openrouter(prompt, model="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
    response = completion(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response

if __name__ == "__main__":
    prompt = "write code for saying hi"
    result = ask_openrouter(prompt)
    print(result)

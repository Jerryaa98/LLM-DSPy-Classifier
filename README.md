
# LLM-based Classification with OpenRouter and DSPy

This project demonstrates a simple, extensible pipeline for LLM-based classification using OpenRouter (via LiteLLM) and DSPy.

## Project Overview

- **Intro Example:** Loads a CSV dataset of simple math addition questions (e.g., "What is 2 + 3?") and uses an LLM to answer and evaluate them. This serves as a minimal working example for LLM-based classification.
- **Extensible Design:** The same pipeline can be adapted to classify any other content, mapping inputs to classes or labels using LLMs.
- Uses OpenRouter LLMs (via LiteLLM) to answer/classify each input.
- Compares the model's answer to the reference/class label in the CSV.
- Prints results and calculates accuracy.


## Structure

- `data/`: Contains datasets, e.g., `math_addition_questions.csv` (math example) or future classification datasets.
- `src/main.py`: Loads the dataset, queries the LLM, and evaluates results.
- `src/generate_math_dataset.py`: Script to generate a CSV of random math addition questions (intro example).


## Setup

1. Create a virtual environment:
	```bash
	python -m venv venv
	source venv/bin/activate  # On Windows: venv\Scripts\activate
	```
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```
3. Create a `.env` file in the project root with your OpenRouter API key:
	```
	OPENROUTER_API_KEY=your_api_key_here
	```


## Usage

1. (Optional) Generate the math addition dataset (intro example):
	```bash
	python src/generate_math_dataset.py
	```
2. Run the main script to answer/classify and evaluate:
	```bash
	python src/main.py
	```

The script will print each question/input, the model's answer/classification, the reference/class label, and whether it was correct. At the end, it prints the overall accuracy.


## Customization & Extending

- To use a different model, edit the `model` argument in `ask_openrouter()` in `src/main.py` or the `model_name` parameter.
- To use your own dataset, place a CSV in the `data/` folder with columns: `question`, `answer`, and optionally `context`.


## Example Output (Math Addition)

```
Q: What is 2 + 3?
OpenRouter Answer: 5
Reference: 5
Correct: True
---
...
Accuracy: 9/10 = 0.90
```


## Roadmap & Extending

- Add more complex question types or classification datasets
- Integrate DSPy for prompt optimization or advanced pipelines.
- Implement more sophisticated evaluation metrics.
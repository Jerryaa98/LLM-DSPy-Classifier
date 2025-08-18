# LLM-based Robust Q&A with Self-Improving Prompts

A test project demonstrating the use of DSPy for building a question-answering system with OpenRouter integration via LiteLLM.

## Project Goal

Build a simple pipeline where an LLM answers questions about provided context using DSPy and OpenRouter, with basic evaluation metrics.

## Structure

1. **Data Utilities**: Simple dataset loading and management functions
2. **DSPy Integration**: Custom integration with OpenRouter via LiteLLM
3. **Evaluation**: Basic metrics for evaluating question answering performance
4. **Main Pipeline**: Orchestration of the Q&A process

## Features

- Question answering on provided context
- Integration with OpenRouter through LiteLLM
- Custom DSPy language model wrapper
- Simple evaluation metrics
- Command-line arguments for configuration

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

Run the main script:

```bash
python src/main.py
```

Additional options:
- `--model` - Specify the OpenRouter model to use
- `--save-data` - Save sample data to CSV
- `--debug` - Print debug information

Example:
```bash
python src/main.py --model openrouter/google/gemma-3-4b-it:free --debug
```

## Extending

To extend this project:
1. Add more complex DSPy modules in `dspy_integration.py`
2. Add your own datasets in `data_utils.py`
3. Implement more sophisticated evaluation metrics in `evaluation.py`
4. Modify the pipeline in `main.py` as needed
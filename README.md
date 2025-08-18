# LLM-based Robust Q&A with Self-Improving Prompts

A test project demonstrating the use of DSPy for building a question-answering system that automatically improves its reasoning through self-optimization.

## Project Goal

Build a small pipeline where an LLM answers questions about a dataset, and DSPy automatically improves its reasoning to reduce mistakes and spurious correlations.

## Structure

1. **Dataset**: Using a small text dataset with some spurious correlations
2. **DSPy Pipeline**:
   - Module 1: Read text/context
   - Module 2: Answer questions
   - Module 3: Evaluate reasoning (DSPy self-optimization)
   - Module 4: Feedback loop for prompt improvement

## Experiments

The project compares:
- Raw LLM with simple prompts
- DSPy pipeline without optimization
- DSPy pipeline with self-optimizing prompts

Metrics include answer accuracy and robustness to spurious cues.
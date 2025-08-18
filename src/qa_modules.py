"""
DSPy modules for the Q&A pipeline.
This file defines the core components of our DSPy-powered question answering system.
"""
import dspy
from typing import List, Dict, Any, Optional

# Define the input/output signature for our Q&A module
class QASignature(dspy.Signature):
    """Signature for question answering with context."""
    context = dspy.InputField(desc="Text passage containing information")
    question = dspy.InputField(desc="Question about the context")
    reasoning = dspy.OutputField(desc="Step-by-step reasoning process")
    answer = dspy.OutputField(desc="Final answer to the question")

# Basic Q&A module without optimization
class BasicQA(dspy.Module):
    """Basic Q&A module that uses an LLM to answer questions based on context."""
    
    def __init__(self):
        super().__init__()
        self.qa_module = dspy.Predict(QASignature)
    
    def forward(self, context: str, question: str) -> Dict[str, str]:
        """Answer a question based on the provided context."""
        system_instruction = (
            "You are a helpful assistant. Answer the following question based on the context. "
            "Respond ONLY in JSON format with the following fields: reasoning, answer.\n"
            "Example:\n"
            "{\n"
            "  \"reasoning\": \"Step-by-step reasoning here.\",\n"
            "  \"answer\": \"Final answer here.\"\n"
            "}"
        )
        context_with_instruction = f"{system_instruction}\nContext: {context}"
        # print(context_with_instruction)
        # input()
        prediction = self.qa_module(context=context_with_instruction, question=question)
        print(prediction)
        input()
        return {
            "reasoning": prediction.reasoning,
            "answer": prediction.answer
        }

# Optimizable Q&A module
class OptimizableQA(dspy.Module):
    """Q&A module designed to be optimized by DSPy."""
    
    def __init__(self):
        super().__init__()
        # Using ChainOfThought to explicitly encourage reasoning
        self.qa_module = dspy.ChainOfThought(QASignature)
    
    def forward(self, context: str, question: str) -> Dict[str, str]:
        """Answer a question based on the provided context with chain of thought reasoning."""
        system_instruction = (
            "You are a helpful assistant. Answer the following question based on the context. "
            "Respond ONLY in JSON with the following fields: reasoning, answer.\n"
            "Example:\n"
            "{\n"
            "  \"reasoning\": \"Step-by-step reasoning here.\",\n"
            "  \"answer\": \"Final answer here.\"\n"
            "}"
        )
        context_with_instruction = f"{system_instruction}\nContext: {context}"
        prediction = self.qa_module(context=context_with_instruction, question=question)
        return {
            "reasoning": prediction.reasoning,
            "answer": prediction.answer
        }

# Evaluator module
class QAEvaluator(dspy.Module):
    """Module to evaluate the quality of Q&A responses."""
    
    def __init__(self):
        super().__init__()
        self.evaluate = dspy.Predict("context, question, reasoning, answer -> score, feedback")
    
    def forward(self, context: str, question: str, reasoning: str, answer: str, reference_answer: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate the quality of a Q&A response."""
        eval_prompt = f"Rate the quality of this question answering response on a scale of 1-10.\n"
        
        if reference_answer:
            eval_prompt += f"Reference answer: {reference_answer}\n"
        
        prediction = self.evaluate(
            context=context, 
            question=question, 
            reasoning=reasoning, 
            answer=answer
        )
        
        return {
            "score": prediction.score,
            "feedback": prediction.feedback
        }

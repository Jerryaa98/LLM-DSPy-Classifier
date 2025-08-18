"""
Integration of DSPy with OpenRouter via LiteLLM.
This module provides a simple interface to use DSPy with OpenRouter.
"""
import os
import dspy
from litellm import completion
from typing import List, Dict, Any

class QASignature(dspy.Signature):
    """
    Signature for question answering module.
    """
    context = dspy.InputField(desc="Text passage to answer questions about")
    question = dspy.InputField(desc="Question to answer based on the context")
    answer = dspy.OutputField(desc="The answer to the question based on the context")

class OpenRouterDSPYModule:
    """
    A simple DSPy module that uses OpenRouter via LiteLLM.
    """
    def __init__(self, model_name="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
        self.model_name = model_name
        # Configure DSPy to use the completion function
        self.lm = self._configure_litellm_for_dspy()
        
    def _configure_litellm_for_dspy(self):
        """
        Configure LiteLLM to work with DSPy.
        This creates a DSPy-compatible language model using LiteLLM.
        """
        # Create a custom LM class that wraps litellm
        class LiteLLMWrapper(dspy.LM):
            def __init__(self, model_name):
                self.model_name = model_name
                self.model = model_name  # For DSPy compatibility
                self.cache = None        # For DSPy compatibility
                self.cache_in_memory = None  # For DSPy compatibility
                self.model_type = 'chat'
                self.kwargs = {
                    "temperature": 0.7,
                    "max_tokens": 1024,
                    "top_p": 1.0,
                    "frequency_penalty": 0.0,
                    "presence_penalty": 0.0
                }
            def basic_request(self, prompt, **kwargs):
                try:
                    response = completion(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=kwargs.get("temperature", self.kwargs["temperature"]),
                        max_tokens=kwargs.get("max_tokens", self.kwargs["max_tokens"])
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"Error calling LiteLLM: {e}")
                    return "Error generating response."
        
        return LiteLLMWrapper(self.model_name)

class QAModule(dspy.Module):
    """
    A simple question answering module using DSPy.
    """
    def __init__(self, lm):
        super().__init__()
        self.lm = lm
        # Use the QASignature class
        self.qa_signature = QASignature
        # Create a predictor using the signature
        self.qa_predictor = dspy.Predict(self.qa_signature)
        
    def forward(self, context: str, question: str) -> Dict[str, Any]:
        """
        Answer a question based on the provided context.
        """
        # Use the predictor to answer the question
        prediction = self.qa_predictor(context=context, question=question)
        return {"answer": prediction.answer}

# Simple function to create a DSPy QA system
def create_qa_system(model_name="openrouter/mistralai/mistral-small-3.1-24b-instruct:free"):
    """
    Create a question answering system using DSPy and OpenRouter.
    """
    openrouter_module = OpenRouterDSPYModule(model_name)
    # Configure DSPy
    dspy.settings.configure(lm=openrouter_module.lm)
    # Create and return the QA module
    return QAModule(openrouter_module.lm)

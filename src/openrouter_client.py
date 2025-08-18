"""
Custom DSPy client for OpenRouter integration.
"""
import os
import json
import dspy
import requests
from typing import Dict, Any, List, Optional, Union

class OpenRouterClient(dspy.LM):
    """
    Custom DSPy client for OpenRouter.
    Extends AsyncLM to provide integration with OpenRouter's API.
    """
    
    def __init__(
        self, 
        api_key: str, 
        model: str, 
        base_url: str = "https://openrouter.ai/api/v1",
        http_referer: str = "http://localhost:3000",
        temperature: float = 0.7,
        max_tokens: int = 1024
    ):
        """
        Initialize the OpenRouter client.
        
        Args:
            api_key: OpenRouter API key
            model: Model identifier (e.g., "anthropic/claude-3-opus:beta")
            base_url: Base URL for OpenRouter API
            http_referer: HTTP referer for API calls
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
        """
        super().__init__(model=model)
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.http_referer = http_referer
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Endpoints
        self.chat_endpoint = f"{self.base_url}/chat/completions"

    def generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        n: int = 1
    ) -> List[str]:
        """
        Generate completions for the given prompt using OpenRouter.
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens if max_tokens is not None else self.max_tokens
        messages = [{"role": "user", "content": prompt}]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.http_referer,
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n
        }
        if stop:
            payload["stop"] = stop
        response = requests.post(self.chat_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        completions = []
        for choice in response_data.get("choices", []):
            if "message" in choice and "content" in choice["message"]:
                completions.append(choice["message"]["content"].strip())
        return completions

    def __call__(self, messages=None, prompt=None, **kwargs):
        # DSPy sometimes calls with messages, sometimes with prompt
        if prompt is None and messages:
            # Convert messages to a single prompt string
            prompt = "\n".join([m["content"] for m in messages if m["role"] == "user"])
        elif prompt is None:
            prompt = ""
        completions = self.generate(prompt, **kwargs)
        return completions[0] if completions else ""

# Function to create an OpenRouter client from config
def create_openrouter_client(
    api_key: str, 
    model: str, 
    base_url: str, 
    http_referer: str
) -> OpenRouterClient:
    """
    Create an OpenRouter client instance from configuration.
    
    Args:
        api_key: OpenRouter API key
        model: Model identifier
        base_url: Base URL for OpenRouter API
        http_referer: HTTP referer for API calls
        
    Returns:
        Configured OpenRouterClient instance
    """
    return OpenRouterClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        http_referer=http_referer
    )

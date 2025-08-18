"""
Configuration file for the DSPy project.
Store API keys and other settings here.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model Configuration
DEFAULT_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"  # OpenRouter model ID
EVALUATION_MODEL = "mistralai/mistral-small-3.1-24b-instruct:free"  # OpenRouter model ID

# OpenRouter Configuration
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_REFERER = "http://localhost:3000"  # Your site URL for OpenRouter referrer
HTTP_REFERER = "http://localhost:3000"  # HTTP referer header

# Dataset paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DATASET_PATH = os.path.join(DATA_DIR, "sports_articles.csv")

# Experiment settings
NUM_EXAMPLES = 50  # Number of examples to use for training/testing
NUM_OPTIMIZATION_STEPS = 3  # Number of optimization steps for DSPy

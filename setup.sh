#!/bin/bash

# Setup script for the DSPy test project

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    # macOS or Linux
    source venv/bin/activate
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    echo "Unsupported OS. Please activate the virtual environment manually."
    exit 1
fi

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Generate dataset
echo "Generating sample dataset..."
python src/data_generator.py

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.template .env
    echo "Please edit the .env file to add your OpenRouter API key."
fi

echo "Setup complete! You can now run the DSPy experiments with 'python src/main.py'"

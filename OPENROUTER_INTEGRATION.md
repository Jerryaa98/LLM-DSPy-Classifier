# OpenRouter Integration for DSPy

This document explains how the project has been updated to use OpenRouter instead of OpenAI's API directly.

## Changes Made

1. **API Key Configuration**
   - Changed from `OPENAI_API_KEY` to `OPENROUTER_API_KEY` in `.env` and `.env.template`
   - Updated configuration in `config.py`

2. **Model IDs**
   - Updated default models to use OpenRouter model identifiers
   - Using `anthropic/claude-3-opus:beta` as the default model

3. **Custom OpenRouter Client**
   - Created a new file `src/openrouter_client.py` with a custom DSPy client for OpenRouter
   - Implemented a wrapper around OpenRouter's API that follows DSPy's interface

4. **Main Script Updates**
   - Updated `main.py` to use the OpenRouter client
   - Added additional configuration parameters (base URL, HTTP referer)

5. **Jupyter Notebook**
   - Added the OpenRouter client implementation to the notebook
   - Updated configuration to use OpenRouter

## Using OpenRouter

OpenRouter provides access to various LLMs through a unified API. To use it:

1. Sign up at [OpenRouter](https://openrouter.ai/)
2. Get your API key from the dashboard
3. Add your API key to the `.env` file:
   ```
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

4. Optionally, you can change the model in `config.py`. Some popular options:
   - `anthropic/claude-3-opus:beta`
   - `anthropic/claude-3-sonnet:beta`
   - `google/gemini-pro`
   - `meta-llama/llama-3-70b-instruct`

## Running the Project

The workflow remains the same as before:

1. Run `./setup.sh` to set up the environment
2. Add your OpenRouter API key to `.env`
3. Run `python src/main.py` or explore the Jupyter notebook

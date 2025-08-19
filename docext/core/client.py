"""
Synchronous API client for vision-language model requests.

This module provides a synchronous client for making API requests to VLM models
with improved error handling and configuration management.
"""

from __future__ import annotations

import os
import requests
from litellm import completion
from loguru import logger
from .config import config


def sync_request(
    messages: list[dict],
    model_name: str = "hosted_vllm/nanonets/Nanonets-OCR-s",
    max_tokens: int = 5000,
    num_completions: int = 1,
    format: dict | None = None,
):
    """
    Make a synchronous request to the VLM model.
    
    Args:
        messages: List of message dictionaries
        model_name: Name of the model to use
        max_tokens: Maximum tokens to generate
        num_completions: Number of completions to generate
        format: Optional format specification
        
    Returns:
        JSON response from the model
        
    Raises:
        ValueError: If required configuration is missing
        requests.RequestException: If API request fails
    """
    try:
        # Get VLM URL from config
        vlm_url = config.get("VLM_MODEL_URL", "")
        
        # Only require VLM_MODEL_URL for hosted_vllm and ollama models
        if (model_name.startswith("hosted_vllm/") or model_name.startswith("ollama/")) and not vlm_url:
            raise ValueError(
                f"VLM_MODEL_URL environment variable is required for model '{model_name}'. "
                "Please set it to the URL of your VLM/OLLAMA server (e.g., 'http://localhost:8000')."
            )
        
        logger.debug(f"Making request to model '{model_name}' at '{vlm_url}'")
        
        completion_args = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "n": num_completions,
            "temperature": 0,
            "api_base": vlm_url
            if model_name.startswith("hosted_vllm/") or model_name.startswith("ollama/")
            else None,
        }

        if model_name.startswith("hosted_vllm/") or model_name.startswith("ollama/"):
            completion_args["api_key"] = config.get("API_KEY", "EMPTY")

        # Only add format argument for Ollama models
        if model_name.startswith("ollama/") and format:
            completion_args["format"] = format
        # elif model_name.startswith("hosted_vllm/") and format: # TODO: Add this back, currently not working in colab
        #     completion_args["guided_json"] = format
        #     if "qwen" in model_name.lower():
        #         completion_args["guided_backend"] = "xgrammar:disable-any-whitespace"
        elif model_name.startswith("openrouter"):
            completion_args["response_format"] = format
        elif "gpt" in model_name.lower():
            # Only set response_format if the prompt mentions "json"
            if any("json" in m.get("text", "").lower() for m in messages if isinstance(m, dict)):
                completion_args["response_format"] = {"type": "json_object"}

        logger.debug(f"Request parameters: {_safe_log_params(completion_args)}")
        
        response = completion(**completion_args)
        logger.debug("Request successful")
        return response.json()
        
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        raise
        
    except Exception as e:
        logger.error(f"Error making request to model '{model_name}': {str(e)}")
        if "Connection refused" in str(e) or "Connection error" in str(e):
            raise requests.RequestException(
                f"Could not connect to model server at {vlm_url}. "
                "Please ensure the server is running and accessible."
            )
        elif "Unauthorized" in str(e) or "401" in str(e):
            raise requests.RequestException(
                "Authentication failed. Please check your API_KEY configuration."
            )
        raise


def _safe_log_params(params: dict) -> dict:
    """Create a safe version of parameters for logging (mask sensitive data)."""
    safe_params = params.copy()
    if 'api_key' in safe_params:
        safe_params['api_key'] = '***masked***'
    return safe_params


def check_model_availability(model_name: str) -> bool:
    """
    Check if a model is available and accessible.
    
    Args:
        model_name: Name of the model to check
        
    Returns:
        bool: True if model is available, False otherwise
    """
    try:
        # Make a simple test request
        test_messages = [{"role": "user", "content": "Hello"}]
        sync_request(test_messages, model_name, max_tokens=1)
        logger.info(f"Model {model_name} is available")
        return True
        
    except Exception as e:
        logger.warning(f"Model {model_name} is not available: {e}")
        return False

"""LLM backend package."""

from polyreflexion.models.base import CallableLLMClient, LLMClient
from polyreflexion.models.ollama import OllamaClient, extract_final_answer
from polyreflexion.models.openai_client import OpenAIClient, parse_json_response
from polyreflexion.models.registry import get_client

__all__ = [
    "CallableLLMClient",
    "LLMClient",
    "OllamaClient",
    "OpenAIClient",
    "extract_final_answer",
    "get_client",
    "parse_json_response",
]

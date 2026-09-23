"""LLM backend package."""

from polyrx.models.base import CallableLLMClient, LLMClient
from polyrx.models.gemini_client import GeminiClient
from polyrx.models.ollama import OllamaClient, extract_final_answer
from polyrx.models.openai_client import OpenAIClient, parse_json_response
from polyrx.models.registry import get_client

__all__ = [
    "CallableLLMClient",
    "GeminiClient",
    "LLMClient",
    "OllamaClient",
    "OpenAIClient",
    "extract_final_answer",
    "get_client",
    "parse_json_response",
]

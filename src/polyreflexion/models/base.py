"""Shared LLM client protocol and adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for text completion backends."""

    def complete(self, prompt: str) -> str: ...


class CallableLLMClient:
    """Adapter wrapping a plain callable as an LLMClient."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def complete(self, prompt: str) -> str:
        return self._fn(prompt)

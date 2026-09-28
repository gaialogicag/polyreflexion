"""Google Gemini LLM backend."""

from __future__ import annotations

from typing import Any

from polyrx.config import GeminiConfig
from polyrx.env import load_env
from polyrx.usage import record_call


def _install_hint() -> str:
    from polyrx.resources import install_extra_hint

    return (
        "The Gemini backend needs the google-genai SDK, which is an optional "
        f"dependency. Install it with: {install_extra_hint('gemini')}"
    )


class GeminiClient:
    """Gemini backend, configured by :class:`GeminiConfig`.

    Mirrors :class:`polyrx.models.openai_client.OpenAIClient`: the config
    carries the *name* of the environment variable holding the key, never the
    key itself, so a resolved config can be written into a run directory
    without redacting anything.

    The SDK is imported inside ``__init__`` rather than at module scope so that
    importing :mod:`polyrx.models` works on an install without the extra. Only
    a run that actually puts a role on Gemini pays for the dependency.
    """

    def __init__(
        self,
        config: GeminiConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        # Load .env lazily: importing this module must not touch the filesystem.
        load_env()
        self.config = config or GeminiConfig()
        self.model = self.config.model
        self.system_prompt = self.config.system_prompt
        self.temperature = self.config.temperature

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover - depends on the install
            raise ImportError(_install_hint()) from exc

        kwargs: dict[str, Any] = {"api_key": api_key or self.config.api_key()}
        if self.config.base_url:
            kwargs["http_options"] = types.HttpOptions(base_url=self.config.base_url)
        self._client = genai.Client(**kwargs)
        self._request_config = self._build_request_config()

    def _build_request_config(self) -> Any:
        """Build the per-request config once; it never varies within a run."""
        from google.genai import types

        settings: dict[str, Any] = {
            "system_instruction": self.system_prompt,
            "temperature": self.temperature,
        }
        if self.config.max_output_tokens is not None:
            settings["max_output_tokens"] = self.config.max_output_tokens
        if self.config.thinking_budget is not None:
            settings["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.config.thinking_budget
            )
        return types.GenerateContentConfig(**settings)

    def complete(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._request_config,
        )
        # Thinking tokens are billed as output, so they belong in the tally even
        # though they never reach `.text`.
        meta = getattr(response, "usage_metadata", None)
        thoughts = getattr(meta, "thoughts_token_count", 0) or 0
        record_call(
            model=self.model,
            provider="gemini",
            prompt_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            completion_tokens=(getattr(meta, "candidates_token_count", 0) or 0) + thoughts,
            cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
            reasoning_tokens=thoughts,
        )
        # `.text` is None when the model returned no text part at all — a safety
        # block, or a thinking budget that consumed the whole output allowance.
        return response.text or ""

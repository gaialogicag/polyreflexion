"""OpenAI API LLM backend."""

from __future__ import annotations

import json
import re

from openai import OpenAI

from polyrx.config import OpenAIConfig
from polyrx.env import load_env
from polyrx.usage import record_call


class OpenAIClient:
    """OpenAI chat-completions backend, configured by :class:`OpenAIConfig`.

    The config carries the *name* of the environment variable holding the key,
    never the key itself, so a resolved config can be written into a run
    directory without redacting anything.
    """

    def __init__(
        self,
        config: OpenAIConfig | None = None,
        *,
        api_key: str | None = None,
    ) -> None:
        # Load .env lazily: importing this module must not touch the filesystem.
        load_env()
        self.config = config or OpenAIConfig()
        self.model = self.config.model
        self.system_prompt = self.config.system_prompt
        self.temperature = self.config.temperature
        self.reasoning_effort = self.config.reasoning_effort
        kwargs: dict = {
            "api_key": api_key or self.config.api_key(),
            "max_retries": self.config.max_retries,
            "timeout": self.config.timeout_s,
        }
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)

    def complete(self, prompt: str) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        if self.reasoning_effort:
            # Precautionary: a reasoning model at non-default effort is
            # assumed to manage its own sampling, so temperature is left out
            # rather than forced alongside it. Not confirmed by testing both
            # together.
            kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["temperature"] = self.temperature
        response = self._client.chat.completions.create(**kwargs)
        # The provider already counted the tokens; throwing the count away is
        # what makes a run's cost unknowable afterwards.
        usage = getattr(response, "usage", None)
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        record_call(
            model=self.model,
            provider="openai",
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            # Both are parts of the totals above, not additions to them.
            cached_tokens=getattr(prompt_details, "cached_tokens", 0) or 0,
            reasoning_tokens=getattr(completion_details, "reasoning_tokens", 0) or 0,
        )
        return response.choices[0].message.content or ""


def parse_json_response(text: str) -> dict:
    """Extract a JSON object from model output."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise

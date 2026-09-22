"""OpenAI API LLM backend."""

from __future__ import annotations

import json
import re

from openai import OpenAI

from polyreflexion.config import OpenAIConfig
from polyreflexion.env import load_env


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
        kwargs: dict = {"api_key": api_key or self.config.api_key()}
        if self.config.base_url:
            kwargs["base_url"] = self.config.base_url
        self._client = OpenAI(**kwargs)

    def complete(self, prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
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

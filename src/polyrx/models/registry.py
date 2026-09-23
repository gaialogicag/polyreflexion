"""Factory turning a backend *role* into a configured client.

Call sites ask for a role — ``"nano"``, ``"judge"``, ``"meta_judge"``,
``"phi"`` — and never name a model. Which model answers a role is a config
decision, which is what makes swapping providers a one-line change in YAML
rather than a search across the codebase. The provider is read off the type of
the role's config, so a run can answer with Gemini and judge with OpenAI.
"""

from __future__ import annotations

from polyrx.config import (
    BackendsConfig,
    GeminiConfig,
    HostedModelConfig,
    OllamaConfig,
    OpenAIConfig,
    PostProcessConfig,
)
from polyrx.models.base import LLMClient
from polyrx.models.gemini_client import GeminiClient
from polyrx.models.ollama import OllamaClient
from polyrx.models.openai_client import OpenAIClient

__all__ = ["ClientRegistry", "active_backends", "get_client", "set_active_backends"]


class ClientRegistry:
    """Builds and caches one client per role for a given backend config.

    Clients are cached because constructing an ``OpenAI`` object opens a
    connection pool; the benchmarks ask for the same role hundreds of times.
    """

    def __init__(
        self,
        backends: BackendsConfig | None = None,
        postprocess: PostProcessConfig | None = None,
    ) -> None:
        self.backends = backends or BackendsConfig()
        self.postprocess = postprocess or PostProcessConfig()
        self._cache: dict[str, LLMClient] = {}

    def get(self, role: str) -> LLMClient:
        """Return the client for a role, building it on first use."""
        if role in self._cache:
            return self._cache[role]
        config = getattr(self.backends, role, None)
        if config is None:
            known = ", ".join(sorted(vars(self.backends)))
            raise ValueError(f"Unknown model role: {role!r}. Use one of: {known}.")
        if isinstance(config, OllamaConfig):
            client: LLMClient = OllamaClient(config, extraction=self.postprocess.extraction)
        # Gemini before OpenAI: both derive from HostedModelConfig, and a bare
        # HostedModelConfig means no provider was selected for this role.
        elif isinstance(config, GeminiConfig):
            client = GeminiClient(config)
        elif isinstance(config, OpenAIConfig):
            client = OpenAIClient(config)
        elif isinstance(config, HostedModelConfig):
            raise TypeError(
                f"Role {role!r} has no provider selected. A backends config must name one "
                f"per hosted role in its defaults, e.g. '- /backends/role@{role}: openai'."
            )
        else:  # pragma: no cover - guarded by the config dataclasses
            raise TypeError(f"Role {role!r} has unsupported config type {type(config).__name__}")
        self._cache[role] = client
        return client

    def describe(self) -> dict[str, str]:
        """Role -> the model identifier that will actually be called."""
        return {role: cfg.model for role, cfg in vars(self.backends).items()}


#: Process-wide registry. Entry points replace it once, at startup, with the
#: registry built from the composed Hydra config; library callers that want
#: isolation construct their own :class:`ClientRegistry` instead.
_active = ClientRegistry()


def set_active_backends(
    backends: BackendsConfig,
    postprocess: PostProcessConfig | None = None,
) -> ClientRegistry:
    """Install the config every later :func:`get_client` call will use."""
    global _active
    _active = ClientRegistry(backends, postprocess)
    return _active


def active_backends() -> ClientRegistry:
    """The registry currently installed."""
    return _active


def get_client(role: str) -> LLMClient:
    """Return the configured client for a role from the active registry."""
    return _active.get(role)

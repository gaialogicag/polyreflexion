"""Swappable meta prompt profiles for different evaluation contexts.

Profiles keep the meta layer object-oriented and changeable:

- ``default`` — general ``meta_default.json`` (judge-feedback aware)
- ``opentom`` — Theory-of-Mind story summaries (``meta_opentom.json``)
- ``gpqa`` — graduate STEM reasoning summaries (``meta_gpqa.json``)
- ``ask`` — interactive Q&A best-answer mode (``meta/ask_prompts.py``)

Benchmark runs select a profile via ``BenchmarkConfig.meta_prompt_profile``;
``polyrx ask`` uses the ``ask`` profile automatically.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from polyreflexion.meta.prompts import META_PROMPTS_PATH, MetaPromptRegistry
from polyreflexion.resources import prompt_path


class MetaPromptProfile(ABC):
    """Abstract profile: produces a configured ``MetaPromptRegistry``."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable profile id stored in benchmark config."""

    @abstractmethod
    def build_registry(self) -> MetaPromptRegistry:
        """Return a fresh registry with this profile's templates."""


class JsonFileMetaProfile(MetaPromptProfile):
    """Profile backed by a JSON template file."""

    def __init__(self, name: str, path: Path) -> None:
        self._name = name
        self._path = path

    @property
    def name(self) -> str:
        return self._name

    def build_registry(self) -> MetaPromptRegistry:
        return MetaPromptRegistry(self._path)


class DefaultMetaProfile(JsonFileMetaProfile):
    """General-purpose meta prompts (``meta_default.json``)."""

    def __init__(self) -> None:
        super().__init__("default", META_PROMPTS_PATH)


class OpenToMMetaProfile(JsonFileMetaProfile):
    """OpenToM-optimized meta prompts for integrated story summaries."""

    def __init__(self) -> None:
        super().__init__("opentom", prompt_path("meta_opentom.json"))


class GPQAMetaProfile(JsonFileMetaProfile):
    """GPQA-optimized meta prompts for scientific reasoning summaries."""

    def __init__(self) -> None:
        super().__init__("gpqa", prompt_path("meta_gpqa.json"))


class AskMetaProfile(MetaPromptProfile):
    """Answer-oriented expand/refine templates for ``polyrx ask``."""

    @property
    def name(self) -> str:
        return "ask"

    def build_registry(self) -> MetaPromptRegistry:
        # Local import avoids circular dependency at module load.
        from polyreflexion.meta.ask_prompts import apply_ask_templates

        return apply_ask_templates(MetaPromptRegistry())


_PROFILES: dict[str, MetaPromptProfile] = {
    "default": DefaultMetaProfile(),
    "opentom": OpenToMMetaProfile(),
    "gpqa": GPQAMetaProfile(),
    "ask": AskMetaProfile(),
}


def get_meta_profile(name: str) -> MetaPromptProfile:
    """Look up a registered profile by name."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_PROFILES))
        raise ValueError(f"Unknown meta prompt profile {name!r}. Known: {known}") from exc


def register_meta_profile(profile: MetaPromptProfile) -> None:
    """Register or replace a profile (useful in tests)."""
    _PROFILES[profile.name] = profile

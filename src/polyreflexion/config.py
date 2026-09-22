"""Structured configuration for every knob in the project.

These dataclasses are the single source of truth.  They are registered with
Hydra's ``ConfigStore`` in :mod:`polyreflexion.conf_store`, which means the YAML
tree under ``polyreflexion/conf/`` is type-checked against them at composition
time — a typo in a config file fails before a single API call is made.

Design rule: **secrets come from the environment, everything else comes from
config**.  A config object therefore never holds an API key, only the name of
the variable to read it from.  That way a resolved config can be written into a
run directory verbatim without leaking credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


@dataclass
class OpenAIConfig:
    """A hosted chat-completions backend (OpenAI or any compatible gateway)."""

    model: str = "gpt-5.4-nano"
    #: Name of the environment variable holding the key — never the key itself.
    api_key_env: str = "OPENAI_API_KEY"
    #: Set for Azure / OpenRouter / vLLM style gateways; ``None`` = OpenAI.
    base_url: str | None = None
    temperature: float = 0.0
    system_prompt: str = "You are an expert in modeling others' mental states."

    def api_key(self) -> str:
        """Read the key from the environment, with an actionable error."""
        key = os.environ.get(self.api_key_env, "")
        if not key:
            raise ValueError(
                f"{self.api_key_env} is not set. Export it or put it in a .env file "
                f"next to your working directory."
            )
        return key


@dataclass
class OllamaConfig:
    """A local Ollama backend."""

    model: str = "phi4-mini-reasoning"
    host: str = "http://127.0.0.1:11434"
    num_ctx: int = 8192
    #: Short generations on purpose: long chains on phi4-mini-reasoning collapse.
    num_predict: int = 512
    timeout_s: int = 1800
    retries: int = 3
    temperature: float = 0.2
    #: ``None`` leaves the flag off the request for older Ollama builds.
    think: bool | None = False
    #: Strip ``<think>`` blocks and return only the final answer.
    strip_reasoning: bool = True


@dataclass
class BackendsConfig:
    """The four named roles the code asks for by alias.

    Keeping the *roles* fixed and the *models* configurable is what lets a
    reader swap in a different provider without touching any call site.
    """

    #: Reflexion engine and meta cycles.
    nano: OpenAIConfig = field(default_factory=OpenAIConfig)
    #: Benchmark label judge (OpenToM labels, ambiguous GPQA answers).
    judge: OpenAIConfig = field(default_factory=lambda: OpenAIConfig(model="gpt-5-mini"))
    #: Polycontextural S/O/D judges — deliberately a different model from
    #: ``judge`` so meta evaluation stays independent of benchmark scoring.
    meta_judge: OpenAIConfig = field(default_factory=lambda: OpenAIConfig(model="gpt-4o"))
    #: Local open-weights backend.
    phi: OllamaConfig = field(default_factory=OllamaConfig)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


@dataclass
class DatasetFile:
    """One file to pull from the Hugging Face Hub, pinned to a commit."""

    repo_id: str
    filename: str
    #: Hub commit sha. ``"main"`` is accepted but makes the run unreproducible,
    #: so :func:`polyreflexion.data.manifest.verify` warns loudly about it.
    revision: str = "main"
    repo_type: str = "dataset"
    #: Expected sha256 of the downloaded file; filled in by ``polyrx-manifest``.
    sha256: str | None = None
    #: Set when the repo is gated and needs an HF token.
    gated: bool = False


@dataclass
class DatasetConfig:
    """A benchmark dataset: a primary source plus optional fallbacks."""

    name: str = "opentom"
    primary: DatasetFile = field(
        default_factory=lambda: DatasetFile(repo_id="SeacowX/OpenToM", filename="opentom.json")
    )
    #: Tried in order when ``primary`` is unavailable (gated, rate-limited).
    fallbacks: list[DatasetFile] = field(default_factory=list)
    #: Extra files (domain label maps, metadata parquets).
    extras: list[DatasetFile] = field(default_factory=list)
    #: Fail the run when a downloaded file does not match its recorded sha256.
    verify_checksums: bool = True


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


@dataclass
class PathsConfig:
    """Filesystem layout, all relative to the original working directory.

    Hydra is configured with ``job.chdir=False``, so these stay stable whether a
    run is launched from the repo root or from an installed console script.
    """

    root: str = "."
    data_dir: str = "data"
    results_dir: str = "results"
    #: Summary / judge-trace cache. Regenerable, always gitignored.
    cache_dir: str = "results/cache"

    def resolved(self, name: str) -> Path:
        """Return one of the directories as an absolute path, creating it."""
        base = Path(self.root).expanduser().resolve()
        path = base / getattr(self, name)
        path.mkdir(parents=True, exist_ok=True)
        return path


# ---------------------------------------------------------------------------
# Engine and meta layer
# ---------------------------------------------------------------------------


@dataclass
class EngineConfig:
    """The recursive reflexion engine itself."""

    #: Packaged prompt template file, or an absolute path to your own.
    prompts: str = "reflexion.json"
    max_workers: int = 8


@dataclass
class MetaLayerConfig:
    """Hydra surface for the polycontextural meta layer.

    The loop's own knobs live in
    :class:`polyreflexion.meta.datatypes.MetaConfig`, which is frozen and used
    inside the controller. This class is the configurable mirror of it plus the
    prompt profile, and :meth:`to_controller` converts between the two. Keeping
    them separate stops Hydra's dataclass handling from leaking into the meta
    layer's own types.
    """

    #: Prompt profile: ``default``, ``opentom``, ``gpqa`` or ``ask``.
    profile: str = "default"
    #: Hard ceiling on observe -> evaluate -> decide cycles.
    max_cycles: int = 2
    #: Reflexion tree depth inside each meta cycle.
    engine_depth: int = 1
    #: Extra depth granted to the single ``F`` reset re-run.
    reset_depth_bonus: int = 1
    #: On meta cycle 0 only, read a fully positive ``(A,W,W)`` as ``(A,W,R)`` so
    #: the loop expands once instead of stopping immediately. Later cycles treat
    #: ``(A,W,W)`` as convergence and terminate.
    reinterpret_full_positive_on_cycle0: bool = True
    #: Stop when successive answers stop resembling the question.
    drift_check_enabled: bool = True
    max_consecutive_drifts: int = 2

    def to_controller(self) -> object:
        """Build the frozen config the meta controller expects."""
        from polyreflexion.meta.datatypes import MetaConfig

        return MetaConfig(
            max_cycles=self.max_cycles,
            engine_depth=self.engine_depth,
            reset_depth_bonus=self.reset_depth_bonus,
            reinterpret_full_positive_on_cycle0=self.reinterpret_full_positive_on_cycle0,
            drift_check_enabled=self.drift_check_enabled,
            max_consecutive_drifts=self.max_consecutive_drifts,
        )


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    """One benchmark run.

    Replaces the old ``BenchmarkConfig`` / ``GPQABenchmarkConfig`` pair — the
    two differed only in the name of the item-count field and the default meta
    profile, both of which now live in the dataset and meta sections.
    """

    #: Which suite to run: ``opentom`` or ``gpqa``.
    suite: str = "opentom"
    num_items: int = 10
    seed: int = 42
    max_workers: int = 4
    #: Condition names resolved against :mod:`polyreflexion.conditions`.
    conditions: list[str] = field(default_factory=list)
    #: Offline path: no network, deterministic stub responses.
    use_stub: bool = False
    #: Isolates the summary cache so a prompt change cannot silently reuse
    #: summaries built by an older prompt version.
    cache_namespace: str = ""
    #: Produce the long-form analysis report alongside the summary report.
    detailed: bool = False
    #: Reuse an existing run: add conditions on the same items.
    merge_from: str | None = None
    #: Reuse an existing run: add new items under the same conditions.
    extend_from: str | None = None
    #: Drop these conditions when merging or extending.
    drop_conditions: list[str] = field(default_factory=list)


@dataclass
class RootConfig:
    """Top-level composed config — what a Hydra entry point receives."""

    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    backends: BackendsConfig = field(default_factory=BackendsConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    meta: MetaLayerConfig = field(default_factory=MetaLayerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

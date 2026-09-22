"""Structured configuration for every knob in the project.

These dataclasses are the single source of truth.  They are registered with
Hydra's ``ConfigStore`` in :mod:`polyrx.conf_store`, which means the YAML
tree under ``conf/`` is type-checked against them at composition
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
    #: Backoff between retries is ``attempt * backoff_step_s``, capped.
    backoff_step_s: int = 5
    backoff_max_s: int = 30
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
    """Where one dataset file comes from.

    Either a local path, or a Hugging Face Hub repo pinned to a commit. A local
    path skips downloading and checksum pinning entirely, which is the right
    thing for data that is not published.
    """

    #: A file on disk, absolute or relative to the working directory. When set,
    #: the Hub fields are ignored.
    path: str | None = None
    repo_id: str = ""
    filename: str = ""
    #: Hub commit sha. ``"main"`` is accepted but makes the run unreproducible,
    #: so :func:`polyrx.data.manifest.verify` warns loudly about it.
    revision: str = "main"
    repo_type: str = "dataset"
    #: Expected sha256 of the downloaded file; filled in by ``polyrx-manifest``.
    sha256: str | None = None
    #: Set when the repo is gated and needs an HF token.
    gated: bool = False


@dataclass
class FieldMap:
    """Which columns or JSON keys carry each part of an item.

    Only the tabular adapter reads this. A dataset whose source needs real
    parsing registers its own adapter and ignores these.
    """

    #: The text to reason over. Falls back to ``question`` when absent, which
    #: is right for self-contained questions.
    context: str | None = None
    question: str = "question"
    answer: str = "answer"
    #: Column holding the allowed answers, if the source states them per row.
    label_space: str | None = None
    #: Column to group by in per-group metrics (a type, a domain, a split).
    group: str | None = None
    #: Columns holding multiple-choice options, in order. When set, the item's
    #: label space becomes the choice letters and the options are rendered into
    #: the prompt.
    choices: list[str] = field(default_factory=list)
    #: Column naming the correct option, when the source gives the text of the
    #: right answer rather than its letter.
    correct_choice: str | None = None
    #: Columns holding the wrong options, when the source separates them.
    incorrect_choices: list[str] = field(default_factory=list)


@dataclass
class DatasetConfig:
    """A benchmark dataset: a primary source plus optional fallbacks.

    ``adapter`` selects how the source is parsed. ``tabular`` handles CSV, TSV,
    JSON and JSONL described by ``fields`` and needs no Python at all; a source
    that needs real parsing registers an adapter under its own name.
    """

    name: str = "dataset"
    #: Registered adapter name. See ``polyrx-datasets list``.
    adapter: str = "tabular"
    #: How the source's columns map onto an item. Tabular adapter only.
    fields: FieldMap = field(default_factory=FieldMap)
    #: Allowed answers when the source does not state them per row. Empty means
    #: the adapter derives them (from the choices, or from the data).
    label_space: str = ""
    #: What the grouping dimension is called in report headings.
    group_name: str = "group"
    #: Whether several items can share one context and must be sampled
    #: together. ``item`` samples each item independently.
    sample_by: str = "item"
    #: Where the data comes from. No default: a dataset config names its own
    #: source, and nothing in this file should know about any particular one.
    primary: DatasetFile = field(default_factory=DatasetFile)
    #: Tried in order when ``primary`` is unavailable (gated, rate-limited).
    fallbacks: list[DatasetFile] = field(default_factory=list)
    #: Extra files (domain label maps, metadata parquets).
    extras: list[DatasetFile] = field(default_factory=list)
    #: Fail the run when a downloaded file does not match its recorded sha256.
    verify_checksums: bool = True
    #: Characters of the content hash that form an item's id. This is a data
    #: format, not a tuning knob: changing it renames every item, which
    #: invalidates every cached summary and makes saved runs unmergeable.
    id_hash_chars: int = 12
    #: Characters of the per-question hash that seeds GPQA choice shuffling.
    #: Changing it reshuffles every question's answer order.
    shuffle_hash_chars: int = 8


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
class ReflexionPrompts:
    """Templates the reflexion engine formats.

    ``{name}`` placeholders are :meth:`str.format` fields; a literal brace is
    written ``{{``. The question-answering and judging templates are
    suite-specific, so only one suite's pair is populated in any given set —
    and neither is present in the interactive set, which has no scoring pass.
    """

    #: Edge label -> template, keyed by the pairs in ``engine._BOUNDARY_KEYS``.
    boundaries: dict[str, str] = field(default_factory=dict)
    #: Perspective letter (``O``/``S``/``B``) -> template.
    perspectives: dict[str, str] = field(default_factory=dict)
    #: Re-integrates the three corner answers into one text.
    summary: str = ""
    #: Asks the question given the built summary. Empty in the interactive set,
    #: which has no scoring pass.
    qa: str = ""
    #: Decides whether an answer matches the gold label when the adapter could
    #: not place it. Empty means fall back to a string comparison.
    judge: str = ""


@dataclass
class MetaPrompts:
    """Templates the polycontextural meta layer formats."""

    judge_subjective: str = ""
    judge_objective: str = ""
    judge_dialectical: str = ""
    #: Turns a positive verdict into a boundary statement for the next cycle.
    boundary_negation: str = ""
    expand_input: str = ""
    refine_input: str = ""
    contradiction_check: str = ""
    drift_check: str = ""


@dataclass
class PromptsConfig:
    """Both template sets, selected as config groups.

    Selecting a set is ``prompts/meta=gpqa``; changing one template is
    ``prompts.meta.judge_objective="..."``; sweeping variants is
    ``-m prompts/meta=default,gpqa``. None of that required code before only
    because the sets were Python classes pointing at packaged JSON files.
    """

    reflexion: ReflexionPrompts = field(default_factory=ReflexionPrompts)
    meta: MetaPrompts = field(default_factory=MetaPrompts)


@dataclass
class EngineConfig:
    """The recursive reflexion engine itself."""

    max_workers: int = 8
    #: Tree depth used when a caller does not specify one. Conditions carry
    #: their own depth, so this only applies outside the benchmark grid.
    default_max_depth: int = 2
    #: The pool is sized to the tree: three branches per node, one level of
    #: lookahead. The cap stops a deep tree from opening thousands of threads.
    worker_pool_cap: int = 128


@dataclass
class MetaLayerConfig:
    """Hydra surface for the polycontextural meta layer.

    The loop's own knobs live in
    :class:`polyrx.meta.datatypes.MetaConfig`, which is frozen and used
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
        from polyrx.meta.datatypes import MetaConfig

        return MetaConfig(
            max_cycles=self.max_cycles,
            engine_depth=self.engine_depth,
            reset_depth_bonus=self.reset_depth_bonus,
            reinterpret_full_positive_on_cycle0=self.reinterpret_full_positive_on_cycle0,
            drift_check_enabled=self.drift_check_enabled,
            max_consecutive_drifts=self.max_consecutive_drifts,
        )


# ---------------------------------------------------------------------------
# Reading answers out of model output
# ---------------------------------------------------------------------------


@dataclass
class DegeneracyConfig:
    """When to treat a model's output as collapsed rather than an answer.

    Small reasoning models sometimes emit token salad or invent a maths puzzle
    instead of answering. A summary that trips this is regenerated once with a
    stricter reminder, so these thresholds change which runs get a second
    attempt — and therefore the published numbers.
    """

    #: Shorter than this (after stripping) is treated as no answer at all.
    min_chars: int = 20
    #: Only the first N characters are examined; collapse shows up early.
    sample_chars: int = 4000
    #: Fraction of digits above which the text reads as numeric salad.
    max_digit_ratio: float = 0.18
    #: Phrases seen in failed phi4-mini-reasoning generations.
    markers: list[str] = field(
        default_factory=lambda: [
            "your name is",
            "named as",
            "segment",
            "\\boxed",
            "the value of r",
            "process results",
        ]
    )
    #: How many distinct markers must appear before the text is rejected. One
    #: marker alone is too easy to hit by chance.
    min_marker_hits: int = 2


@dataclass
class AnswerExtractionConfig:
    """Pulling a final answer out of a reasoning model's full response."""

    #: A ``**bold**`` span longer than this is prose, not a label.
    max_bold_chars: int = 80
    #: A bold span with more words than this is a heading, not a label.
    max_bold_words: int = 8


@dataclass
class PostProcessConfig:
    """Everything applied to raw model text before it is scored."""

    degeneracy: DegeneracyConfig = field(default_factory=DegeneracyConfig)
    extraction: AnswerExtractionConfig = field(default_factory=AnswerExtractionConfig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass
class ReportConfig:
    """Presentation only: nothing here changes a score."""

    #: Figure size in inches for the single-series charts.
    chart_width: float = 10.0
    chart_height: float = 5.0
    #: The grouped by-question-type chart needs more room.
    grouped_chart_width: float = 12.0
    grouped_chart_height: float = 6.0
    #: Item identifiers are content hashes; this many characters is enough to
    #: tell rows apart in a table.
    id_display_chars: int = 8
    #: Question text is truncated to keep the table readable.
    question_display_chars: int = 80
    #: Marker excerpts quoted in the detailed report.
    excerpt_chars: int = 100
    #: Question text in the narrower flip-examples table.
    flip_question_chars: int = 60
    #: Rows shown per flip-examples table in the detailed report.
    max_flip_examples: int = 12
    #: Example excerpts listed per marker section.
    max_marker_examples: int = 8
    #: Print a progress line every N answered question/condition pairs.
    progress_every: int = 10


# ---------------------------------------------------------------------------
# Provenance collection
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceConfig:
    """What the run record gathers about its own environment."""

    #: Packages whose version can change a result. A full ``pip freeze`` buries
    #: the handful that matter.
    tracked_packages: list[str] = field(
        default_factory=lambda: ["openai", "huggingface_hub", "hydra-core", "omegaconf"]
    )
    #: Seconds to wait on each git subprocess. Provenance must never be the
    #: reason a run hangs.
    git_timeout_s: int = 5
    #: Characters of a commit or revision shown in human-readable messages.
    short_hash_chars: int = 8
    #: Bytes read per chunk when hashing a dataset file.
    hash_chunk_bytes: int = 1048576


# ---------------------------------------------------------------------------
# The experiment grid
# ---------------------------------------------------------------------------


@dataclass
class ConditionSpec:
    """One cell of the experiment grid, as it appears in ``conf/conditions``."""

    name: str = "?"
    backend: str = "nano"
    #: Reflexion tree depth. ``0`` answers the item with no reflexion pass.
    depth: int = 0
    #: Meta loop budget. ``0`` disables the meta layer. A meta condition runs a
    #: depth-1 reflexion on cycle 0, so it carries ``depth >= 1``.
    meta_cycles: int = 0
    #: Retired name kept working for a release; points at its replacement.
    alias_of: str | None = None
    note: str = ""


@dataclass
class ConditionsConfig:
    """The whole grid. Order here is report column order."""

    conditions: list[ConditionSpec] = field(default_factory=list)
    #: Fail a run when the grid no longer matches ``data/conditions.lock.json``.
    #: Turning this off lets a run produce numbers that cannot be compared with
    #: any previous one, so leave it on unless you are re-locking deliberately.
    enforce_lock: bool = True


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

    num_items: int = 10
    seed: int = 42
    max_workers: int = 4
    #: Condition names resolved against :mod:`polyrx.conditions`.
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
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    conditions: ConditionsConfig = field(default_factory=ConditionsConfig)
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    provenance: ProvenanceConfig = field(default_factory=ProvenanceConfig)
    meta: MetaLayerConfig = field(default_factory=MetaLayerConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

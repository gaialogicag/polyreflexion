"""Everything a reader needs to decide whether they can reproduce a run.

A results file that records only the experiment settings is not reproducible:
the same ``seed=42`` with a silently updated dataset, a re-pointed model alias
or an edited prompt template produces different numbers and no way to tell.
This module collects the rest — code version, resolved model identifiers,
dataset revisions, prompt hashes — into a block stored next to every run.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path

from polyrx import __version__

#: Packages whose version can change a result. Kept short on purpose — a full
#: ``pip freeze`` buries the three that matter.
_TRACKED_PACKAGES = ("openai", "huggingface_hub", "hydra-core", "omegaconf")


def file_sha256(path: Path) -> str:
    """Stream a file through sha256; works on datasets too large to hold in RAM."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_templates(prompts: object, prefix: str = "") -> dict[str, str]:
    """Flatten a prompt config into ``dotted.name -> sha256``."""
    out: dict[str, str] = {}
    if isinstance(prompts, dict):
        items = list(prompts.items())
    elif hasattr(prompts, "__dict__"):
        items = list(vars(prompts).items())
    else:
        return out
    for key, value in items:
        name = f"{prefix}{key}"
        if isinstance(value, str):
            out[name] = text_sha256(value)
        else:
            out.update(_hash_templates(value, f"{name}."))
    return out


def _git(*args: str) -> str | None:
    """Run a git command in the package's repository, or return ``None``.

    Returns ``None`` for an installed wheel, a tarball download, or any
    environment without git — all legitimate ways to run this code.
    """
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


@dataclass
class GitInfo:
    """Which commit produced the run."""

    commit: str | None = None
    branch: str | None = None
    #: True when the working tree had uncommitted changes. A dirty run is not
    #: reproducible from the commit alone, so reports say so explicitly.
    dirty: bool = False

    @classmethod
    def collect(cls) -> GitInfo:
        commit = _git("rev-parse", "HEAD")
        if commit is None:
            return cls()
        return cls(
            commit=commit,
            branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
            dirty=bool(_git("status", "--porcelain")),
        )


@dataclass
class ModelInfo:
    """The model actually sent to the provider, not the alias asked for."""

    role: str
    backend: str
    model: str
    base_url: str | None = None
    temperature: float | None = None


@dataclass
class DatasetInfo:
    """Which bytes the run scored against."""

    name: str
    repo_id: str
    filename: str
    revision: str
    sha256: str | None = None
    #: Content-hashed item identifiers actually evaluated. Lets a reader re-run
    #: the exact sample without trusting the sampler to behave identically.
    item_ids: list[str] = field(default_factory=list)


@dataclass
class UsageInfo:
    """What the run cost, so a reader can budget a replication."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_calls: int = 0
    estimated_usd: float | None = None
    wall_clock_s: float | None = None


@dataclass
class Provenance:
    """The full provenance block written into every run JSON."""

    polyreflexion_version: str = __version__
    recorded_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    git: GitInfo = field(default_factory=GitInfo)
    python: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    packages: dict[str, str] = field(default_factory=dict)
    models: list[ModelInfo] = field(default_factory=list)
    datasets: list[DatasetInfo] = field(default_factory=list)
    #: Dotted template name -> sha256, e.g. ``reflexion.summary``. Prompts are
    #: configuration now, so the run's config carries the text; this is what
    #: makes a change to it detectable at a glance.
    prompts: dict[str, str] = field(default_factory=dict)
    usage: UsageInfo = field(default_factory=UsageInfo)

    @classmethod
    def collect(
        cls,
        *,
        models: list[ModelInfo] | None = None,
        datasets: list[DatasetInfo] | None = None,
        prompts: object | None = None,
    ) -> Provenance:
        """Gather everything available in this process.

        ``prompts`` is the resolved :class:`~polyrx.config.PromptsConfig`.
        Each template is hashed on its own, so comparing two runs names the
        template that changed rather than only reporting that the set differs.
        """
        packages: dict[str, str] = {}
        for name in _TRACKED_PACKAGES:
            try:
                packages[name] = _pkg_version(name)
            except PackageNotFoundError:
                continue
        return cls(
            git=GitInfo.collect(),
            packages=packages,
            models=models or [],
            datasets=datasets or [],
            prompts=_hash_templates(prompts) if prompts is not None else {},
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def warnings(self) -> list[str]:
        """Reasons this run is not exactly reproducible, in plain sentences."""
        issues: list[str] = []
        if self.git.commit is None:
            issues.append("No git commit recorded — the code version is unknown.")
        elif self.git.dirty:
            issues.append(
                f"Working tree was dirty at {self.git.commit[:8]} — "
                "the commit does not describe the code that ran."
            )
        for dataset in self.datasets:
            if dataset.revision in {"main", "master", ""}:
                issues.append(
                    f"Dataset {dataset.name} was pulled from {dataset.revision or 'an unpinned ref'!r}; "
                    "upstream can change it. Pin a commit sha."
                )
            if dataset.sha256 is None:
                issues.append(f"Dataset {dataset.name} has no recorded checksum.")
        return issues

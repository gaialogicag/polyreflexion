"""Download datasets at a pinned revision and verify the bytes.

The previous loaders called ``hf_hub_download`` with no ``revision``, which
resolves to whatever ``main`` points at today.  When an upstream dataset is
corrected — GPQA has been revised, OpenToM has been re-uploaded — every
published accuracy number silently stops matching the data, with nothing in the
results file to show it.

Two mechanisms fix that:

* **Revision pinning.** Every fetch names a Hub commit sha.
* **A manifest.** ``data/MANIFEST.json`` records the sha256 of each file as it
  was when the published results were produced. A mismatch stops the run.

Neither redistributes the data. GPQA is gated and license-restricted; OpenToM
has its own terms. Users still download from the original source.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from polyrx.config import DatasetConfig, DatasetFile
from polyrx.provenance import DatasetInfo, file_sha256

MANIFEST_NAME = "MANIFEST.json"

#: Refs that do not identify a fixed snapshot.
_FLOATING_REFS = frozenset({"", "main", "master", "HEAD"})


class DatasetVerificationError(RuntimeError):
    """A downloaded file does not match its recorded checksum."""


@dataclass
class ManifestEntry:
    """One recorded file: where it came from and what it hashed to."""

    repo_id: str
    filename: str
    revision: str
    sha256: str
    repo_type: str = "dataset"
    size_bytes: int = 0

    @property
    def key(self) -> str:
        return f"{self.repo_type}:{self.repo_id}/{self.filename}"


@dataclass
class Manifest:
    """The set of dataset files a published result was computed from."""

    entries: dict[str, ManifestEntry] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> Manifest:
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(entries={k: ManifestEntry(**v) for k, v in raw.get("entries", {}).items()})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "_comment": (
                "Checksums of the dataset files the published results were computed from. "
                "Regenerate with: polyrx-manifest pin"
            ),
            "entries": {k: asdict(v) for k, v in sorted(self.entries.items())},
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def record(self, entry: ManifestEntry) -> None:
        self.entries[entry.key] = entry

    def expected(self, spec: DatasetFile) -> ManifestEntry | None:
        return self.entries.get(f"{spec.repo_type}:{spec.repo_id}/{spec.filename}")


class DatasetFetcher:
    """Fetch dataset files, honouring pins and checksums.

    Parameters
    ----------
    cache_dir:
        Where downloads land. One subdirectory per dataset name.
    verify:
        Raise on checksum mismatch. Turn off only when deliberately updating to
        a new dataset revision, then re-pin the manifest.
    """

    def __init__(self, cache_dir: Path, *, verify: bool = True) -> None:
        self.cache_dir = Path(cache_dir)
        self.verify = verify
        self.manifest = Manifest.load(self.cache_dir / MANIFEST_NAME)
        #: Populated as files are fetched; handed to the run's provenance block.
        self.fetched: list[DatasetInfo] = []

    # -- single file --------------------------------------------------------

    def fetch(self, spec: DatasetFile, *, subdir: str | None = None) -> Path:
        """Download one file and verify it, returning the local path."""
        from huggingface_hub import hf_hub_download

        target_dir = self.cache_dir / subdir if subdir else self.cache_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        if spec.revision in _FLOATING_REFS:
            print(
                f"WARNING: {spec.repo_id}/{spec.filename} is pinned to "
                f"{spec.revision or 'no revision'!r}. Upstream can change it under you. "
                f"Run 'polyrx-manifest pin' to record a fixed commit."
            )

        path = Path(
            hf_hub_download(
                spec.repo_id,
                spec.filename,
                repo_type=spec.repo_type,
                revision=spec.revision or None,
                local_dir=str(target_dir),
            )
        )
        self._verify(spec, path)
        return path

    def _verify(self, spec: DatasetFile, path: Path) -> None:
        digest = file_sha256(path)
        # An explicit sha256 on the spec wins; otherwise fall back to whatever
        # the manifest recorded for this file.
        recorded = self.manifest.expected(spec)
        expected = spec.sha256 or (recorded.sha256 if recorded is not None else None)
        if expected and digest != expected:
            message = (
                f"{spec.repo_id}/{spec.filename} does not match the recorded checksum.\n"
                f"  expected sha256 {expected}\n"
                f"  actual   sha256 {digest}\n"
                f"The upstream file changed, or the pin is wrong. Results computed from "
                f"these bytes are not comparable to the published numbers."
            )
            if self.verify:
                raise DatasetVerificationError(message)
            print(f"WARNING: {message}")

        self.fetched.append(
            DatasetInfo(
                name=spec.repo_id.split("/")[-1],
                repo_id=spec.repo_id,
                filename=spec.filename,
                revision=spec.revision,
                sha256=digest,
            )
        )

    # -- a configured dataset, with fallbacks -------------------------------

    def fetch_dataset(self, config: DatasetConfig) -> Path:
        """Fetch ``config.primary``, falling back in order when it is unavailable.

        A gated repository the user has no token for is an expected outcome,
        not an error — that is what the mirrors in ``fallbacks`` are for.
        """
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

        attempts: list[tuple[DatasetFile, str]] = []
        for spec in (config.primary, *config.fallbacks):
            try:
                return self.fetch(spec, subdir=config.name)
            except (GatedRepoError, HfHubHTTPError, OSError) as exc:
                attempts.append((spec, f"{type(exc).__name__}: {exc}"))
                print(f"  {spec.repo_id}/{spec.filename} unavailable ({type(exc).__name__})")

        detail = "\n".join(f"  - {s.repo_id}/{s.filename}: {why}" for s, why in attempts)
        raise RuntimeError(
            f"Could not obtain dataset {config.name!r} from any configured source:\n{detail}\n"
            f"If the primary source is gated, set HF_TOKEN and accept the dataset terms."
        )

    # -- pinning ------------------------------------------------------------

    def pin(self, config: DatasetConfig) -> Manifest:
        """Resolve every configured file to a concrete commit and record its hash.

        Run this once when adopting a dataset, and again deliberately when
        moving to a newer revision. The resulting manifest is what makes a
        published number checkable by someone else.
        """
        from huggingface_hub import HfApi

        api = HfApi()
        for spec in (config.primary, *config.fallbacks, *config.extras):
            try:
                info = api.repo_info(spec.repo_id, repo_type=spec.repo_type)
            except Exception as exc:
                print(f"  skip {spec.repo_id}: {type(exc).__name__}: {exc}")
                continue
            resolved = DatasetFile(
                repo_id=spec.repo_id,
                filename=spec.filename,
                revision=info.sha or spec.revision,
                repo_type=spec.repo_type,
                gated=spec.gated,
            )
            try:
                path = self.fetch(resolved, subdir=config.name)
            except Exception as exc:
                print(f"  skip {spec.repo_id}/{spec.filename}: {type(exc).__name__}: {exc}")
                continue
            self.manifest.record(
                ManifestEntry(
                    repo_id=resolved.repo_id,
                    filename=resolved.filename,
                    revision=resolved.revision,
                    sha256=file_sha256(path),
                    repo_type=resolved.repo_type,
                    size_bytes=path.stat().st_size,
                )
            )
            print(f"  pinned {resolved.repo_id}/{resolved.filename} @ {resolved.revision[:8]}")
        self.manifest.save(self.cache_dir / MANIFEST_NAME)
        return self.manifest

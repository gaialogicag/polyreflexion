"""The dataset interface every benchmark goes through.

A benchmark item is the same shape whatever it came from: some text to reason
about, a question, a gold answer, and the set of answers that count. Datasets
differ in four things and only four:

1. how the source file parses into items,
2. what the allowed answers are,
3. how a free-form model response maps onto one of them,
4. which field groups items for per-group metrics.

An adapter supplies those. Everything downstream — summary building, caching,
meta cycles, merging, metrics, reports — is written once against :class:`Item`
and never learns which dataset it is looking at.

Most datasets need no adapter at all: the tabular adapter in
:mod:`polyrx.datasets.tabular` is driven entirely from ``conf/dataset/*.yaml``.
Write a class only when the source format needs real parsing.
"""

from __future__ import annotations

import hashlib
import random
from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from polyrx.config import DatasetConfig

__all__ = [
    "DatasetAdapter",
    "Item",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "sample_by_group",
]


@dataclass(frozen=True)
class Item:
    """One scoreable unit, whatever dataset it came from."""

    #: Content hash. Stable across runs, which is what lets a cached summary and
    #: a saved run still refer to the same item months later.
    item_id: str
    #: The text the reflexion tree reasons over. For a story-based dataset this
    #: is the narrative; for a self-contained question it is the question itself.
    context: str
    #: What the question-answering pass is shown, including any answer choices.
    prompt_text: str
    #: The question on its own, for report tables.
    question: str
    gold_label: str
    #: Comma-separated allowed answers, e.g. ``"A, B, C, D"`` or ``"Yes, No"``.
    label_space: str
    #: How items are grouped in per-group metrics: a question type, a domain, a
    #: split. ``"unspecified"`` when the dataset offers nothing to group by.
    group: str = "unspecified"
    #: Anything the adapter wants to carry through to reports.
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def content_id(*parts: str, id_chars: int = 12) -> str:
        """Content hash for an item, from whatever uniquely identifies it.

        Content-addressed rather than positional so that re-sampling, merging
        and extending a run all refer to the same item.
        """
        blob = "\n".join(parts)
        return hashlib.sha256(blob.encode()).hexdigest()[:id_chars]


class DatasetAdapter(ABC):
    """Turns a configured source into :class:`Item` objects and scores answers."""

    #: Name used in ``conf/dataset/*.yaml`` to select this adapter.
    name: str = ""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    # -- loading ------------------------------------------------------------

    @abstractmethod
    def load(self, path: Path) -> list[Item]:
        """Parse the downloaded source file into items."""

    # -- scoring ------------------------------------------------------------

    @abstractmethod
    def match_label(self, prediction: str, item: Item) -> str:
        """Map a free-form model response onto one of ``item.label_space``.

        Return the raw prediction when nothing matches; the judge decides what
        to do with an unmatched answer.
        """

    def needs_judge(self, matched: str, item: Item) -> bool:
        """Whether an answer needs the language-model judge rather than a
        string comparison. Default: only when matching failed to land on an
        allowed label."""
        allowed = {part.strip().casefold() for part in item.label_space.split(",")}
        return matched.strip().casefold() not in allowed

    # -- grouping -----------------------------------------------------------

    def group_label(self) -> str:
        """Human-readable name of the grouping dimension, for report headings."""
        return self.config.group_name or "group"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTERS: dict[str, type[DatasetAdapter]] = {}


def register_adapter(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
    """Register an adapter under its ``name``. Usable as a decorator."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a name to be registered")
    _ADAPTERS[cls.name] = cls
    return cls


def get_adapter(config: DatasetConfig) -> DatasetAdapter:
    """Instantiate the adapter named by ``config.adapter``."""
    # Import for side effects: the built-in adapters register on import.
    from polyrx.datasets import gpqa, opentom, tabular  # noqa: F401

    try:
        cls = _ADAPTERS[config.adapter]
    except KeyError as exc:
        known = ", ".join(sorted(_ADAPTERS))
        raise ValueError(
            f"Unknown dataset adapter {config.adapter!r} for dataset "
            f"{config.name!r}. Known adapters: {known}. "
            f"Most sources need no adapter of their own — try adapter: tabular "
            f"and describe the columns in conf/dataset/{config.name}.yaml."
        ) from exc
    return cls(config)


def available_adapters() -> tuple[str, ...]:
    from polyrx.datasets import gpqa, opentom, tabular  # noqa: F401

    return tuple(sorted(_ADAPTERS))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_by_group(
    items: Iterable[Item],
    *,
    count: int,
    seed: int,
    exclude: set[str] | None = None,
    only: set[str] | None = None,
    group_key: str = "context",
) -> list[Item]:
    """Sample ``count`` groups of items, returning every item in each.

    Datasets where several questions share one passage must be sampled by
    passage, not by question, or the same passage appears under several
    conditions and the summary cache is built more times than necessary.
    ``group_key`` picks what a "passage" means: ``"context"`` groups questions
    sharing a text, ``"item"`` treats every item as its own group.
    """
    buckets: dict[str, list[Item]] = {}
    for item in items:
        key = item.item_id if group_key == "item" else Item.content_id(item.context)
        buckets.setdefault(key, []).append(item)

    if only is not None:
        chosen = sorted(k for k in buckets if any(i.item_id in only for i in buckets[k]))
        return [i for k in chosen for i in buckets[k] if i.item_id in only]

    candidates = sorted(buckets)
    if exclude:
        candidates = [k for k in candidates if not any(i.item_id in exclude for i in buckets[k])]
    picked = random.Random(seed).sample(candidates, min(count, len(candidates)))
    return [item for key in picked for item in buckets[key]]

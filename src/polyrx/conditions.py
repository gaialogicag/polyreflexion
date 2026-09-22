"""Experimental conditions as data.

A *condition* is one cell of the experiment grid: which backend produces the
reasoning, how deep the reflexion tree goes, and how many meta cycles are
allowed on top.  Everything else about a condition — its cache key, whether it
needs a summary pass, which budget it continues from — is derived from those
three numbers.

The grid itself is **not** in this file. It lives in ``conf/conditions/``, like
every other setting, and is loaded into a :class:`ConditionRegistry`. This
module only defines what a condition *is* and how its derived properties work.

Two files, two jobs:

* ``conf/conditions/*.yaml`` — the live grid. Edit it to add a condition.
* ``data/conditions.lock.json`` — a frozen record of the grid the published
  results were produced with, in the same spirit as ``data/MANIFEST.json`` for
  datasets. :func:`check_against_lock` compares the two, because a changed
  cache key does not crash: it silently orphans every cached summary.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

__all__ = [
    "LOCK_FILENAME",
    "Condition",
    "ConditionRegistry",
    "GridLockError",
    "check_against_lock",
    "registry_from_config",
    "require_lock_match",
    "write_lock",
]


@dataclass(frozen=True)
class Condition:
    """One experimental condition.

    Parameters
    ----------
    name:
        Stable identifier used in configs, reports and result JSON.  Renaming
        one invalidates published results, so treat it as part of the data
        format rather than as a label.
    backend:
        Alias resolved by :func:`polyrx.models.registry.get_client`.
    depth:
        Reflexion tree depth. ``0`` means the model answers the raw item with
        no reflexion pass at all (the ``*_direct`` baselines).
    meta_cycles:
        Budget for the polycontextural meta loop. ``0`` disables the meta layer.
        Cycle 0 of a meta run is always a depth-1 reflexion on the original
        text, which is why meta conditions carry ``depth=1``.
    """

    name: str
    backend: str
    depth: int = 0
    meta_cycles: int = 0
    #: Set on retired names kept working for one release; points at the
    #: replacement so callers can warn once and move on.
    alias_of: str | None = None
    #: Free-text note surfaced in generated reports.
    note: str = ""

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(f"{self.name}: depth must be >= 0, got {self.depth}")
        if self.meta_cycles < 0:
            raise ValueError(f"{self.name}: meta_cycles must be >= 0, got {self.meta_cycles}")
        if self.meta_cycles and not self.depth:
            raise ValueError(
                f"{self.name}: a meta condition runs a reflexion pass on cycle 0, "
                f"so depth must be >= 1 (got {self.depth})"
            )

    # -- derived properties -------------------------------------------------

    @property
    def is_direct(self) -> bool:
        """True when the model answers the item with no intermediate summary."""
        return self.depth == 0 and self.meta_cycles == 0

    @property
    def is_meta(self) -> bool:
        return self.meta_cycles > 0

    @property
    def is_reflexion(self) -> bool:
        """A plain reflexion condition: a tree, but no meta layer above it."""
        return self.depth > 0 and self.meta_cycles == 0

    @property
    def uses_summary(self) -> bool:
        """Whether the question-answering pass sees a summary instead of the item."""
        return not self.is_direct

    @property
    def summary_key(self) -> str:
        """Cache key for this condition's summaries.

        Distinct depths and meta budgets must never share a cache entry — that
        is how a "depth 3" number silently becomes a depth 1 number.
        """
        if self.is_direct:
            return ""
        if self.is_meta:
            return f"{self.backend}_meta_c{self.meta_cycles}"
        return self.backend if self.depth == 1 else f"{self.backend}_d{self.depth}"

    @property
    def prior_summary_key(self) -> str | None:
        """Cache key of the next-smaller meta budget, when one exists.

        Budget *N* continues from budget *N-1*'s summary when it is cached,
        which is what makes the incremental sweeps cheap.  Budget 1 has no
        predecessor: its cycle 0 is a fresh depth-1 reflexion.
        """
        if not self.is_meta or self.meta_cycles <= 1:
            return None
        return f"{self.backend}_meta_c{self.meta_cycles - 1}"

    @property
    def family(self) -> str:
        """Group used for report ordering: ``nano_direct`` -> ``nano``."""
        return self.backend

    def label(self) -> str:
        """Human-readable description for report tables."""
        if self.is_direct:
            return f"{self.backend}, no reflexion"
        if self.is_meta:
            cycles = "cycle" if self.meta_cycles == 1 else "cycles"
            return f"{self.backend}, depth {self.depth} + {self.meta_cycles} meta {cycles}"
        return f"{self.backend}, reflexion depth {self.depth}"


class ConditionRegistry:
    """Lookup and ordering for the known conditions.

    Ordering is the registration order, which is also the report column order.
    Keeping it explicit beats sorting by name, where ``nano_meta_c10`` would
    land between ``c1`` and ``c2``.
    """

    def __init__(self, conditions: list[Condition] | None = None) -> None:
        self._by_name: dict[str, Condition] = {}
        for condition in conditions or []:
            self.register(condition)

    def register(self, condition: Condition) -> None:
        """Add or replace a condition."""
        self._by_name[condition.name] = condition

    def get(self, name: str) -> Condition:
        """Resolve a name, following one level of alias."""
        try:
            condition = self._by_name[name]
        except KeyError as exc:
            known = ", ".join(self.names())
            raise KeyError(f"Unknown condition {name!r}. Known: {known}") from exc
        if condition.alias_of:
            target = self._by_name[condition.alias_of]
            # Keep the alias's own name so reports show what the user asked for
            # while the behaviour comes entirely from the target.
            return replace(target, name=condition.name, note=condition.note)
        return condition

    def resolve_all(self, names: list[str] | tuple[str, ...]) -> tuple[Condition, ...]:
        """Resolve a list of names, preserving registry order, deduplicated."""
        wanted = {self.get(name).name for name in names}
        return tuple(c for c in self.ordered() if c.name in wanted)

    def ordered(self) -> tuple[Condition, ...]:
        """Every registered condition, in registration order."""
        return tuple(self.get(name) for name in self._by_name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._by_name)

    def expand_group(self, group: str) -> tuple[str, ...]:
        """Expand a shorthand group name used by the command line.

        ``nano`` means every nano condition; ``nano_nod3`` is the same without
        the expensive depth-3 run.
        """
        if group == "all":
            return self.names()
        if group.endswith("_nod3"):
            backend = group.removesuffix("_nod3")
            return tuple(c.name for c in self.ordered() if c.backend == backend and c.depth != 3)
        return tuple(c.name for c in self.ordered() if c.backend == group)


# ---------------------------------------------------------------------------
# Building a registry from configuration
# ---------------------------------------------------------------------------


def registry_from_config(conditions: Iterable[Any]) -> ConditionRegistry:
    """Build a registry from the composed ``conditions`` config group.

    Accepts anything with the Condition fields as attributes or keys, so it
    works with the structured config, a plain mapping, or Condition instances.
    """
    specs: list[Condition] = []
    for entry in conditions:
        if isinstance(entry, Condition):
            specs.append(entry)
            continue

        def get(key: str, default: Any = None, item: Any = entry) -> Any:
            """Read a field whether the entry is a mapping or an object."""
            return item.get(key, default) if isinstance(item, dict) else getattr(item, key, default)

        name, backend = get("name"), get("backend")
        if not name or not backend:
            raise ValueError(
                f"Condition entry needs both a name and a backend, got {entry!r}. "
                f"Check conf/conditions/."
            )
        specs.append(
            Condition(
                name=str(name),
                backend=str(backend),
                depth=get("depth", 0) or 0,
                meta_cycles=get("meta_cycles", 0) or 0,
                alias_of=get("alias_of", None) or None,
                note=get("note", "") or "",
            )
        )
    return ConditionRegistry(specs)


# ---------------------------------------------------------------------------
# The grid lock
# ---------------------------------------------------------------------------

#: Default location of the lock, relative to the repository root.
LOCK_FILENAME = "conditions.lock.json"

#: Fields compared between the live grid and the lock. These are exactly the
#: values that used to live in six parallel dictionaries, and every cached
#: summary on disk is keyed by ``summary_key``.
_LOCKED_FIELDS = ("backend", "depth", "meta_cycles", "summary_key", "prior_summary_key")


class GridLockError(RuntimeError):
    """The live condition grid no longer matches the locked one."""


def _lock_row(condition: Condition) -> dict[str, Any]:
    return {field: getattr(condition, field) for field in _LOCKED_FIELDS}


def write_lock(registry: ConditionRegistry, path: Path) -> Path:
    """Record the current grid as the reference for future runs."""
    payload = {
        "_comment": (
            "The condition grid the published results were produced with. Every cached "
            "summary is keyed by summary_key, so a change here orphans the cache. "
            "Regenerate deliberately with: polyrx-conditions lock"
        ),
        "conditions": {name: _lock_row(registry.get(name)) for name in registry.names()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def check_against_lock(registry: ConditionRegistry, path: Path) -> list[str]:
    """Return the differences between the live grid and the locked one.

    An empty list means they agree. A missing lock file returns no differences:
    a fresh checkout with no published results has nothing to protect yet.
    """
    if not path.is_file():
        return []
    locked = json.loads(path.read_text(encoding="utf-8")).get("conditions", {})

    problems: list[str] = []
    for name in sorted(set(locked) - set(registry.names())):
        problems.append(f"{name}: in the lock but no longer in the grid")
    for name, row in sorted(locked.items()):
        if name not in registry.names():
            continue
        live = _lock_row(registry.get(name))
        for field in _LOCKED_FIELDS:
            if row.get(field) != live[field]:
                problems.append(f"{name}.{field}: locked {row.get(field)!r}, now {live[field]!r}")
    return problems


def require_lock_match(registry: ConditionRegistry, path: Path) -> None:
    """Raise unless the live grid matches the lock."""
    problems = check_against_lock(registry, path)
    if problems:
        raise GridLockError(
            "The condition grid no longer matches "
            + str(path)
            + ":\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nEvery cached summary is keyed by summary_key, so a mismatch orphans the "
            "cache and makes existing runs unmergeable. If the change is deliberate, "
            "re-lock with `polyrx-conditions lock` in the same commit and say which "
            "published results it invalidates."
        )

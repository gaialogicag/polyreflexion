"""Experimental conditions as data, not as six parallel dictionaries.

A *condition* is one cell of the experiment grid: which backend produces the
reasoning, how deep the reflexion tree goes, and how many meta cycles are
allowed on top.  Everything else about a condition — its cache key, whether it
needs a summary pass, which budget it continues from — is derived from those
three numbers rather than looked up in yet another table.

Before this module the same information lived in ``CONDITIONS``,
``BACKEND_BY_CONDITION``, ``DEPTH_BY_CONDITION``, ``META_CYCLES_BY_CONDITION``,
``SUMMARY_KEY_BY_CONDITION`` and ``PRIOR_META_SUMMARY_KEY``.  Adding one
condition meant editing five of them consistently.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

__all__ = [
    "DEFAULT_CONDITIONS",
    "Condition",
    "ConditionRegistry",
    "default_registry",
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
# Fidelity check against the pre-refactor lookup tables
# ---------------------------------------------------------------------------

#: The published grid exactly as the six dictionaries in the original
#: ``benchmark/runner.py`` spelled it, recovered from commit c9a64a1.
#:
#: Each row is ``(backend, depth, meta_cycles, summary_key, prior_summary_key)``
#: where ``None`` means the original table had no entry for that condition.
#:
#: This exists because the derivation in :class:`Condition` replaced those
#: tables, and a derivation that is wrong by one character does not crash: it
#: silently rebuilds summaries instead of reusing them, and lets a merge pair
#: one condition's predictions with another condition's summaries. The failure
#: is wrong numbers, not an error, so it has to be checked rather than noticed.
_PUBLISHED_GRID: dict[str, tuple[str, int | None, int | None, str | None, str | None]] = {
    "nano_direct": ("nano", None, None, None, None),
    "nano_reflexion": ("nano", 1, None, "nano", None),
    "nano_reflexion_d2": ("nano", 2, None, "nano_d2", None),
    "nano_reflexion_d3": ("nano", 3, None, "nano_d3", None),
    "nano_meta_c1": ("nano", 1, 1, "nano_meta_c1", None),
    "nano_meta_c2": ("nano", 1, 2, "nano_meta_c2", "nano_meta_c1"),
    "nano_meta_c3": ("nano", 1, 3, "nano_meta_c3", "nano_meta_c2"),
    "nano_meta_c4": ("nano", 1, 4, "nano_meta_c4", "nano_meta_c3"),
    # Legacy alias: the original tables gave it nano_meta_c2's budget and cache.
    "nano_meta": ("nano", 1, 2, "nano_meta_c2", "nano_meta_c1"),
    "phi_direct": ("phi", None, None, None, None),
    "phi_reflexion": ("phi", 1, None, "phi", None),
    "phi_reflexion_d2": ("phi", 2, None, "phi_d2", None),
}


def check_published_grid(registry: ConditionRegistry) -> None:
    """Raise if the derived grid no longer matches the published one.

    Called when the default registry is built, so a change that would silently
    orphan an existing summary cache fails at import instead of at the point
    where the numbers come out wrong.
    """
    problems: list[str] = []

    missing = set(_PUBLISHED_GRID) - set(registry.names())
    if missing:
        problems.append(f"conditions dropped from the published grid: {sorted(missing)}")

    for name, (backend, depth, cycles, summary_key, prior_key) in _PUBLISHED_GRID.items():
        if name not in registry.names():
            continue
        spec = registry.get(name)
        # The old tables simply omitted direct conditions, so absent means 0.
        for label, expected, actual in (
            ("backend", backend, spec.backend),
            ("depth", depth or 0, spec.depth),
            ("meta_cycles", cycles or 0, spec.meta_cycles),
            ("summary_key", summary_key or "", spec.summary_key),
            ("prior_summary_key", prior_key, spec.prior_summary_key),
        ):
            if expected != actual:
                problems.append(f"{name}.{label}: published {expected!r}, derived {actual!r}")

    if problems:
        raise AssertionError(
            "The derived condition grid no longer matches the published one.\n"
            + "\n".join(f"  - {p}" for p in problems)
            + "\n\nEvery cached summary is keyed by summary_key, so a mismatch orphans "
            "the cache and makes existing runs unmergeable. If the change is "
            "deliberate, update _PUBLISHED_GRID in the same commit and say in the "
            "message which published results it invalidates."
        )


def _build_default() -> ConditionRegistry:
    """The published condition grid.

    Generated from the grid definition rather than written out, so a new meta
    budget is one number, not five dictionary entries.
    """
    conditions: list[Condition] = [
        Condition("nano_direct", backend="nano"),
        Condition("nano_reflexion", backend="nano", depth=1),
        Condition("nano_reflexion_d2", backend="nano", depth=2),
        Condition("nano_reflexion_d3", backend="nano", depth=3),
    ]
    conditions += [
        Condition(f"nano_meta_c{n}", backend="nano", depth=1, meta_cycles=n) for n in (1, 2, 3, 4)
    ]
    conditions += [
        Condition(
            "nano_meta",
            backend="nano",
            depth=1,
            meta_cycles=2,
            alias_of="nano_meta_c2",
            note="Legacy alias kept for older result files.",
        ),
        Condition("phi_direct", backend="phi"),
        Condition("phi_reflexion", backend="phi", depth=1),
        Condition("phi_reflexion_d2", backend="phi", depth=2),
    ]
    registry = ConditionRegistry(conditions)
    check_published_grid(registry)
    return registry


#: The registry the benchmarks use unless a caller supplies its own.
default_registry = _build_default()

#: Canonical display / run order, kept as plain strings for config files.
DEFAULT_CONDITIONS: tuple[str, ...] = default_registry.names()

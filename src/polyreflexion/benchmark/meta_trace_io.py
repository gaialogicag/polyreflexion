"""Save and load lightweight meta-cycle traces for benchmark attribution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from polyreflexion.meta.datatypes import Dimension, MetaResult


@dataclass(frozen=True)
class TraceJudgment:
    dimension: Dimension
    positive: bool
    label: str
    rationale: str
    method: str


@dataclass(frozen=True)
class TraceCycle:
    index: int
    triple: str
    reinterpreted: bool
    global_value: str
    action: str
    judgments: tuple[TraceJudgment, ...]


@dataclass(frozen=True)
class MetaTraceRecord:
    """Minimal cycle history for dimension attribution (no full engine tree)."""

    selected_cycle: int
    termination_reason: str
    cycles: tuple[TraceCycle, ...]

    def selected(self) -> TraceCycle | None:
        if not self.cycles or self.selected_cycle < 0:
            return None
        for cycle in self.cycles:
            if cycle.index == self.selected_cycle:
                return cycle
        return self.cycles[-1]

    def first(self) -> TraceCycle | None:
        return self.cycles[0] if self.cycles else None


def meta_trace_path(summary_cache_path: Path) -> Path:
    """Trace JSON lives beside the cached summary ``.txt`` file."""
    return summary_cache_path.with_suffix(".meta.json")


def save_meta_trace(result: MetaResult, path: Path) -> None:
    """Persist judge triples per cycle for offline attribution."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "selected_cycle": result.selected_cycle,
        "termination_reason": result.termination_reason,
        "cycles": [
            {
                "index": cycle.index,
                "triple": cycle.evaluation.as_triple(),
                "reinterpreted": cycle.evaluation.reinterpreted,
                "global_value": cycle.global_value.value,
                "action": cycle.action,
                "judgments": [j.to_dict() for j in cycle.evaluation.judgments()],
            }
            for cycle in result.cycles
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_meta_trace(path: Path) -> MetaTraceRecord | None:
    """Load a trace written by ``save_meta_trace``; return ``None`` if missing."""
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    cycles: list[TraceCycle] = []
    for raw in payload.get("cycles", []):
        judgments = tuple(
            TraceJudgment(
                dimension=Dimension(j["dimension"]),
                positive=bool(j["positive"]),
                label=str(j["label"]),
                rationale=str(j.get("rationale", "")),
                method=str(j.get("method", "")),
            )
            for j in raw.get("judgments", [])
        )
        cycles.append(
            TraceCycle(
                index=int(raw["index"]),
                triple=str(raw.get("triple", "")),
                reinterpreted=bool(raw.get("reinterpreted", False)),
                global_value=str(raw.get("global_value", "")),
                action=str(raw.get("action", "")),
                judgments=judgments,
            )
        )
    return MetaTraceRecord(
        selected_cycle=int(payload.get("selected_cycle", -1)),
        termination_reason=str(payload.get("termination_reason", "")),
        cycles=tuple(cycles),
    )


def judgment_for(cycle: TraceCycle, dimension: Dimension) -> TraceJudgment | None:
    for judgment in cycle.judgments:
        if judgment.dimension is dimension:
            return judgment
    return None

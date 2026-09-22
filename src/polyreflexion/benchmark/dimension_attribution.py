"""Attribute QA repairs to polycontextural judge dimension changes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from polyreflexion.benchmark.meta_trace_io import MetaTraceRecord, judgment_for
from polyreflexion.benchmark.qualitative_analysis import FlipRecord, collect_flips
from polyreflexion.meta.datatypes import DIMENSION_ORDER, NEGATIVE_LABEL, POSITIVE_LABEL, Dimension


@dataclass(frozen=True)
class LabelTransition:
    """One dimension's label change between two meta cycles."""

    dimension: Dimension
    from_label: str
    to_label: str
    from_positive: bool
    to_positive: bool

    @property
    def pole_pair(self) -> str:
        """Display form, e.g. ``F→A``, ``R→W``."""
        return f"{self.from_label}→{self.to_label}"


@dataclass
class DimensionAttributionReport:
    """Story-level judge changes linked to QA repairs."""

    baseline: str
    condition: str
    summary_key: str
    repaired_questions: int
    repaired_stories: int
    stories_with_trace: int
    stories_without_trace: int
    stories_no_dimension_repair: int
    repair_flips: Counter[str] = field(default_factory=Counter)  # dimension.value
    all_transitions: Counter[str] = field(default_factory=Counter)  # "dialectical:R→W"
    repair_flips_by_story: dict[str, tuple[str, ...]] = field(default_factory=dict)
    example_trajectories: list[dict] = field(default_factory=list)


def transitions_between(
    first: MetaTraceRecord,
    last: MetaTraceRecord | None = None,
) -> list[LabelTransition]:
    """Label changes from the first meta cycle to the selected final cycle."""
    start = first.first()
    end = (last or first).selected() or first.cycles[-1] if first.cycles else None
    if start is None or end is None:
        return []
    transitions: list[LabelTransition] = []
    for dimension in DIMENSION_ORDER:
        j0 = judgment_for(start, dimension)
        j1 = judgment_for(end, dimension)
        if j0 is None or j1 is None or j0.label == j1.label:
            continue
        transitions.append(
            LabelTransition(
                dimension=dimension,
                from_label=j0.label,
                to_label=j1.label,
                from_positive=j0.positive,
                to_positive=j1.positive,
            )
        )
    return transitions


def repair_dimensions(transitions: list[LabelTransition]) -> tuple[str, ...]:
    """Dimensions that flipped from negative to positive (judge repair)."""
    return tuple(t.dimension.value for t in transitions if not t.from_positive and t.to_positive)


def repaired_story_ids(
    results: list[dict],
    *,
    baseline: str,
    condition: str,
) -> dict[str, list[FlipRecord]]:
    """Map story_id -> gain flips (baseline wrong, condition correct)."""
    flip = collect_flips(results, baseline=baseline, condition=condition)
    by_story: dict[str, list[FlipRecord]] = defaultdict(list)
    for record in flip.gains:
        by_story[record.story_id].append(record)
    return dict(by_story)


def build_dimension_attribution(
    results: list[dict],
    traces: dict[str, MetaTraceRecord],
    *,
    baseline: str,
    condition: str,
    summary_key: str,
    max_examples: int = 5,
) -> DimensionAttributionReport:
    """Link QA wrong→right flips to judge dimension repairs in meta traces."""
    by_story = repaired_story_ids(results, baseline=baseline, condition=condition)
    report = DimensionAttributionReport(
        baseline=baseline,
        condition=condition,
        summary_key=summary_key,
        repaired_questions=sum(len(v) for v in by_story.values()),
        repaired_stories=len(by_story),
        stories_with_trace=0,
        stories_without_trace=0,
        stories_no_dimension_repair=0,
    )

    for story_id in sorted(by_story):
        trace = traces.get(story_id)
        if trace is None or not trace.cycles:
            report.stories_without_trace += 1
            continue
        report.stories_with_trace += 1
        transitions = transitions_between(trace)
        for transition in transitions:
            key = f"{transition.dimension.value}:{transition.pole_pair}"
            report.all_transitions[key] += 1
        dims = repair_dimensions(transitions)
        if not dims:
            report.stories_no_dimension_repair += 1
        else:
            report.repair_flips_by_story[story_id] = dims
            for dim in dims:
                report.repair_flips[dim] += 1

        if len(report.example_trajectories) < max_examples and trace.cycles:
            report.example_trajectories.append(
                {
                    "story_id": story_id,
                    "repaired_count": len(by_story[story_id]),
                    "trajectory": " → ".join(c.triple for c in trace.cycles),
                    "termination": trace.termination_reason,
                    "repair_dims": ", ".join(dims) or "(none)",
                }
            )

    return report


def repair_dimension_shares(report: DimensionAttributionReport) -> dict[str, float]:
    """Normalised shares of negative→positive flips per dimension."""
    total = sum(report.repair_flips.values())
    if total == 0:
        return {dim.value: 0.0 for dim in DIMENSION_ORDER}
    return {dim.value: report.repair_flips[dim.value] / total for dim in DIMENSION_ORDER}


def format_dimension_table(report: DimensionAttributionReport) -> list[str]:
    """Markdown lines for the dimension repair share table."""
    shares = repair_dimension_shares(report)
    total_flips = sum(report.repair_flips.values())
    lines = [
        "| Dimension | Pole repair | Flips | Share |",
        "|-----------|-------------|------:|------:|",
    ]
    labels = {
        Dimension.SUBJECTIVE: f"{NEGATIVE_LABEL[Dimension.SUBJECTIVE]}→{POSITIVE_LABEL[Dimension.SUBJECTIVE]}",
        Dimension.OBJECTIVE: f"{NEGATIVE_LABEL[Dimension.OBJECTIVE]}→{POSITIVE_LABEL[Dimension.OBJECTIVE]}",
        Dimension.DIALECTICAL: f"{NEGATIVE_LABEL[Dimension.DIALECTICAL]}→{POSITIVE_LABEL[Dimension.DIALECTICAL]}",
    }
    for dimension in DIMENSION_ORDER:
        count = report.repair_flips[dimension.value]
        lines.append(
            f"| {dimension.value.capitalize()} | {labels[dimension]} | {count} | "
            f"{shares[dimension.value]:.0%} |"
        )
    lines.append("")
    lines.append(
        f"*Total dimension repair credits: {total_flips} "
        f"(stories can contribute multiple dimensions).*"
    )
    lines.append("")
    return lines


def format_transition_table(report: DimensionAttributionReport) -> list[str]:
    """Markdown table of all label transitions cycle 0 → final."""
    if not report.all_transitions:
        return ["*(no transitions recorded)*", ""]
    lines = [
        "| Dimension | Transition | Count |",
        "|-----------|------------|------:|",
    ]
    for key, count in report.all_transitions.most_common():
        dim, transition = key.split(":", 1)
        lines.append(f"| {dim} | {transition} | {count} |")
    lines.append("")
    return lines

"""Aggregate metrics for OpenToM benchmark runs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class ConditionMetrics:
    """Scores for one experimental condition."""

    condition: str
    accuracy: float
    total: int
    correct: int
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)


def compute_metrics(results: list[dict]) -> dict[str, ConditionMetrics]:
    """Compute accuracy per condition and question type."""
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        for condition in row.get("predictions", {}):
            if condition not in row.get("judgments", {}):
                continue
            by_condition[condition].append(
                {
                    "correct": row["judgments"][condition]["correct"],
                    "question_type": row["question_type"],
                }
            )

    metrics: dict[str, ConditionMetrics] = {}
    for condition, rows in by_condition.items():
        if not rows:
            continue
        correct_flags = [r["correct"] for r in rows]
        accuracy = sum(correct_flags) / len(correct_flags)

        by_type: dict[str, list[bool]] = defaultdict(list)
        for r in rows:
            by_type[r["question_type"]].append(r["correct"])

        type_scores = {
            qtype: {
                "accuracy": sum(flags) / len(flags),
                "count": len(flags),
            }
            for qtype, flags in by_type.items()
        }

        metrics[condition] = ConditionMetrics(
            condition=condition,
            accuracy=accuracy,
            total=len(rows),
            correct=sum(correct_flags),
            by_type=type_scores,
        )
    return metrics

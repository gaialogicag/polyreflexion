"""Aggregate scoring.

``compute_metrics`` turns the per-item rows a run produces into the accuracy
numbers that get published, so an error here is an error in the paper.
"""

from __future__ import annotations

import pytest

from polyrx.benchmark.metrics import ConditionMetrics, compute_metrics


def row(group: str, **judged: bool) -> dict:
    """One result row: a prediction and a judgment per condition."""
    return {
        "group": group,
        "predictions": {condition: "answer" for condition in judged},
        "judgments": {condition: {"correct": correct} for condition, correct in judged.items()},
    }


class TestAccuracy:
    def test_accuracy_counts_correct_over_total(self) -> None:
        metrics = compute_metrics(
            [
                row("basic", direct=True),
                row("basic", direct=True),
                row("basic", direct=False),
                row("basic", direct=False),
            ]
        )

        assert metrics["direct"].accuracy == 0.5
        assert metrics["direct"].correct == 2
        assert metrics["direct"].total == 4

    def test_all_correct_and_all_wrong_are_the_endpoints(self) -> None:
        perfect = compute_metrics([row("basic", direct=True)])
        hopeless = compute_metrics([row("basic", direct=False)])

        assert perfect["direct"].accuracy == 1.0
        assert hopeless["direct"].accuracy == 0.0

    def test_conditions_are_scored_independently(self) -> None:
        metrics = compute_metrics(
            [
                row("basic", direct=True, reflexion=False),
                row("basic", direct=True, reflexion=False),
            ]
        )

        assert metrics["direct"].accuracy == 1.0
        assert metrics["reflexion"].accuracy == 0.0

    def test_returns_a_condition_metrics_instance(self) -> None:
        metrics = compute_metrics([row("basic", direct=True)])

        assert isinstance(metrics["direct"], ConditionMetrics)
        assert metrics["direct"].condition == "direct"


class TestGroupBreakdown:
    def test_by_type_splits_accuracy_per_group(self) -> None:
        metrics = compute_metrics(
            [
                row("first-order", direct=True),
                row("first-order", direct=True),
                row("second-order", direct=True),
                row("second-order", direct=False),
            ]
        )

        by_type = metrics["direct"].by_type
        assert by_type["first-order"] == {"accuracy": 1.0, "count": 2}
        assert by_type["second-order"] == {"accuracy": 0.5, "count": 2}

    def test_group_counts_sum_to_the_condition_total(self) -> None:
        metrics = compute_metrics(
            [row("a", direct=True), row("b", direct=False), row("b", direct=True)]
        )
        scored = metrics["direct"]

        assert sum(entry["count"] for entry in scored.by_type.values()) == scored.total


class TestPartialRows:
    def test_a_prediction_without_a_judgment_is_not_scored(self) -> None:
        """An unjudged prediction must not be counted as wrong. Scoring it
        would silently depress the accuracy of whichever condition failed to
        judge, which is the condition under test."""
        unjudged = row("basic", direct=True)
        unjudged["predictions"]["reflexion"] = "answer"

        metrics = compute_metrics([unjudged])

        assert "reflexion" not in metrics
        assert metrics["direct"].total == 1

    def test_a_row_with_no_predictions_contributes_nothing(self) -> None:
        metrics = compute_metrics([{"group": "basic", "predictions": {}, "judgments": {}}])

        assert metrics == {}

    def test_rows_missing_the_prediction_and_judgment_keys_are_tolerated(self) -> None:
        metrics = compute_metrics([{"group": "basic"}, row("basic", direct=True)])

        assert metrics["direct"].total == 1


class TestEmptyInput:
    def test_no_rows_yields_no_conditions(self) -> None:
        assert compute_metrics([]) == {}


class TestRequiredFields:
    def test_a_scored_row_without_a_group_raises(self) -> None:
        """Documents current behaviour: ``predictions`` and ``judgments`` are
        read defensively, ``group`` is not."""
        headless = {"predictions": {"direct": "answer"}, "judgments": {"direct": {"correct": True}}}

        with pytest.raises(KeyError):
            compute_metrics([headless])

"""Valuation triple -> global value, and the geometry it selects.

Both modules are pure: no LLM, no configuration, no IO. The rule is a lookup on
how many judges were positive, plus one exception on the first meta cycle.
"""

from __future__ import annotations

import pytest

from polyrx.engine import Perspective
from polyrx.meta.datatypes import (
    ContextJudgment,
    Dimension,
    GlobalValue,
    PolyEvaluation,
    summary_hash,
)
from polyrx.meta.geometry import GeometryMapper
from polyrx.meta.interpretation import Interpreter


def evaluation(subjective: bool, objective: bool, dialectical: bool) -> PolyEvaluation:
    return PolyEvaluation(
        subjective=ContextJudgment.make(Dimension.SUBJECTIVE, subjective, "because"),
        objective=ContextJudgment.make(Dimension.OBJECTIVE, objective, "because"),
        dialectical=ContextJudgment.make(Dimension.DIALECTICAL, dialectical, "because"),
    )


class TestGlobalValue:
    @pytest.mark.parametrize(
        ("verdicts", "expected"),
        [
            pytest.param((True, True, True), GlobalValue.W, id="three-converged"),
            pytest.param((True, True, False), GlobalValue.R, id="two-expand"),
            pytest.param((True, False, False), GlobalValue.A, id="one-refine"),
            pytest.param((False, False, False), GlobalValue.F, id="none-reset"),
        ],
    )
    def test_the_lookup_counts_positives(
        self, verdicts: tuple[bool, bool, bool], expected: GlobalValue
    ) -> None:
        value, _ = Interpreter().interpret(evaluation(*verdicts), cycle_index=1)

        assert value is expected

    def test_which_judges_were_positive_does_not_matter(self) -> None:
        """Only the count drives the rule."""
        first, _ = Interpreter().interpret(evaluation(True, True, False), cycle_index=1)
        second, _ = Interpreter().interpret(evaluation(False, True, True), cycle_index=1)

        assert first is second is GlobalValue.R

    def test_the_lookup_can_be_replaced(self) -> None:
        """Kept as data so a future meta level can override it."""
        interpreter = Interpreter(
            lookup={3: GlobalValue.F, 2: GlobalValue.F, 1: GlobalValue.F, 0: GlobalValue.F}
        )

        value, _ = interpreter.interpret(evaluation(True, True, False), cycle_index=1)

        assert value is GlobalValue.F


class TestAntiConvergenceRule:
    def test_all_positive_on_cycle_zero_is_flipped_to_expansion(self) -> None:
        """The initial reflexion looking fully positive is assumed dialectical
        exhaustion, not convergence, so the loop expands instead of stopping."""
        value, reinterpreted = Interpreter().interpret(evaluation(True, True, True), cycle_index=0)

        assert value is GlobalValue.R
        assert reinterpreted.reinterpreted
        assert not reinterpreted.dialectical.positive

    def test_the_flip_is_recorded_as_a_reinterpretation(self) -> None:
        _, reinterpreted = Interpreter().interpret(evaluation(True, True, True), cycle_index=0)

        assert reinterpreted.dialectical.method == "reinterpretation"
        assert "cycle 0" in reinterpreted.dialectical.rationale

    def test_a_later_cycle_is_left_alone(self) -> None:
        """By then meta has steered something, so three positives are genuine."""
        value, unchanged = Interpreter().interpret(evaluation(True, True, True), cycle_index=1)

        assert value is GlobalValue.W
        assert not unchanged.reinterpreted

    def test_a_partly_positive_cycle_zero_is_left_alone(self) -> None:
        value, unchanged = Interpreter().interpret(evaluation(True, True, False), cycle_index=0)

        assert value is GlobalValue.R
        assert not unchanged.reinterpreted

    def test_the_rule_can_be_switched_off(self) -> None:
        interpreter = Interpreter(reinterpret_full_positive_on_cycle0=False)

        value, unchanged = interpreter.interpret(evaluation(True, True, True), cycle_index=0)

        assert value is GlobalValue.W
        assert not unchanged.reinterpreted

    def test_the_original_evaluation_is_not_mutated(self) -> None:
        original = evaluation(True, True, True)

        Interpreter().interpret(original, cycle_index=0)

        assert original.dialectical.positive
        assert not original.reinterpreted


class TestGeometry:
    def test_two_positives_expand_across_their_edge(self) -> None:
        state = GeometryMapper().map(
            GlobalValue.R, frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE})
        )

        assert state.canonical.kind == "expand_edge"
        assert len(state.canonical.anchor) == 2

    def test_the_edge_is_the_vertices_of_the_positive_dimensions(self) -> None:
        state = GeometryMapper().map(
            GlobalValue.R, frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE})
        )

        assert set(state.canonical.anchor) == {
            Perspective.SUBJECTIVITY,
            Perspective.OBJECTIVITY,
        }

    def test_one_positive_descends_into_its_corner(self) -> None:
        state = GeometryMapper().map(GlobalValue.A, frozenset({Dimension.DIALECTICAL}))

        assert state.canonical.kind == "descend_corner"
        assert state.canonical.anchor == (Perspective.CONCEPTUALITY,)

    def test_convergence_rests(self) -> None:
        state = GeometryMapper().map(GlobalValue.W, frozenset(Dimension))

        assert state.canonical.kind == "rest"
        assert state.canonical.anchor == ()

    def test_failure_returns_to_the_root(self) -> None:
        state = GeometryMapper().map(GlobalValue.F, frozenset())

        assert state.canonical.kind == "reset_root"

    def test_the_mapping_is_deterministic(self) -> None:
        positives = frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE})

        first = GeometryMapper().map(GlobalValue.R, positives)
        second = GeometryMapper().map(GlobalValue.R, positives)

        assert first.canonical == second.canonical
        assert first.surplus == second.surplus

    def test_the_edge_order_does_not_depend_on_set_iteration_order(self) -> None:
        """A frozenset has no order. The anchor must still come out the same,
        or the descent path recorded in a trace is not reproducible."""
        one = GeometryMapper().map(
            GlobalValue.R, frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE})
        )
        other = GeometryMapper().map(
            GlobalValue.R, frozenset({Dimension.OBJECTIVE, Dimension.SUBJECTIVE})
        )

        assert one.canonical.anchor == other.canonical.anchor


class TestSurplus:
    def test_every_move_carries_one(self) -> None:
        """Preserved on every state even though nothing consumes it yet."""
        cases = [
            (GlobalValue.R, frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE})),
            (GlobalValue.A, frozenset({Dimension.SUBJECTIVE})),
            (GlobalValue.W, frozenset(Dimension)),
            (GlobalValue.F, frozenset()),
        ]
        for value, positives in cases:
            assert GeometryMapper().map(value, positives).surplus is not None

    def test_the_surplus_differs_from_the_canonical_arrangement(self) -> None:
        state = GeometryMapper().map(
            GlobalValue.R, frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE})
        )

        assert state.surplus != state.canonical

    def test_convergence_records_the_expansion_it_did_not_take(self) -> None:
        state = GeometryMapper().map(GlobalValue.W, frozenset(Dimension))

        assert state.surplus.kind == "expand_full"
        assert len(state.surplus.anchor) == 3


class TestDescentPath:
    def test_expansion_appends_the_edge(self) -> None:
        state = GeometryMapper().map(
            GlobalValue.R, frozenset({Dimension.SUBJECTIVE, Dimension.OBJECTIVE}), path=("S",)
        )

        assert len(state.path) == 2
        assert state.path[0] == "S"

    def test_refinement_appends_the_apex(self) -> None:
        state = GeometryMapper().map(GlobalValue.A, frozenset({Dimension.DIALECTICAL}), path=("S",))

        assert state.path == ("S", Perspective.CONCEPTUALITY.value)

    def test_convergence_leaves_the_path_alone(self) -> None:
        state = GeometryMapper().map(GlobalValue.W, frozenset(Dimension), path=("S", "O"))

        assert state.path == ("S", "O")

    def test_failure_clears_the_path(self) -> None:
        """A reset goes back to the root triangle, so the descent so far is gone."""
        state = GeometryMapper().map(GlobalValue.F, frozenset(), path=("S", "O"))

        assert state.path == ()


class TestSummaryHash:
    def test_the_same_text_hashes_the_same(self) -> None:
        assert summary_hash("the answer") == summary_hash("the answer")

    def test_whitespace_and_case_are_normalised(self) -> None:
        """Oscillation detection compares summaries across cycles; a reflowed
        line is not a different answer."""
        assert summary_hash("The  Answer\nis here") == summary_hash("the answer is here")

    def test_different_text_hashes_differently(self) -> None:
        assert summary_hash("yes") != summary_hash("no")


class TestEvaluationHelpers:
    def test_the_display_triple_uses_the_pole_labels(self) -> None:
        assert evaluation(True, True, True).as_triple() == "(A,W,W)"

    def test_a_failing_dialectical_judge_reads_as_redundant(self) -> None:
        assert evaluation(True, True, False).as_triple() == "(A,W,R)"

    def test_positives_come_back_in_canonical_order(self) -> None:
        positives = evaluation(True, False, True).positives()

        assert positives == (Dimension.SUBJECTIVE, Dimension.DIALECTICAL)

    def test_repair_priorities_name_only_the_failing_judges(self) -> None:
        priorities = evaluation(True, False, False).repair_priorities()

        assert "objective" in priorities
        assert "dialectical" in priorities
        assert "subjective" not in priorities

    def test_repair_priorities_say_so_when_everything_passed(self) -> None:
        assert "All judges passed" in evaluation(True, True, True).repair_priorities()

    def test_feedback_marks_what_must_be_fixed(self) -> None:
        feedback = evaluation(True, False, True).feedback_block()

        assert "NOT satisfied" in feedback
        assert "preserve this strength" in feedback

    def test_guidance_for_an_absent_dimension_raises(self) -> None:
        with pytest.raises(ValueError, match="No judgment"):
            evaluation(True, True, True).guidance_for("not-a-dimension")  # type: ignore[arg-type]

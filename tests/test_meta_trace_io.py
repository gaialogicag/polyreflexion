"""Meta-cycle traces on disk.

The trace is what dimension attribution reads months after a run, so the
round trip has to survive without the engine trees that produced it.
"""

from __future__ import annotations

from pathlib import Path

from polyrx.benchmark.meta_trace_io import (
    judgment_for,
    load_meta_trace,
    meta_trace_path,
    save_meta_trace,
)
from polyrx.engine import Node, Perspective, ReflexionResult, Region
from polyrx.meta.datatypes import (
    ContextJudgment,
    Dimension,
    GeometricArrangement,
    GlobalValue,
    MetaCycle,
    MetaResult,
    PolyEvaluation,
    TopologyState,
)


def evaluation(subjective: bool, objective: bool, dialectical: bool) -> PolyEvaluation:
    return PolyEvaluation(
        subjective=ContextJudgment.make(Dimension.SUBJECTIVE, subjective, "s-rationale"),
        objective=ContextJudgment.make(Dimension.OBJECTIVE, objective, "o-rationale"),
        dialectical=ContextJudgment.make(Dimension.DIALECTICAL, dialectical, "d-rationale"),
    )


def reflexion_result(summary: str) -> ReflexionResult:
    leaf = Node(region=Region.root(), depth=0, perspectives=dict.fromkeys(Perspective, "x"))
    return ReflexionResult(
        tree=leaf, corner_answers=dict.fromkeys(Perspective, "x"), summary=summary
    )


def cycle(index: int, verdicts: tuple[bool, bool, bool], action: str) -> MetaCycle:
    return MetaCycle(
        index=index,
        input_text="the story",
        all_inputs=("the story",),
        result=reflexion_result(f"summary {index}"),
        evaluation=evaluation(*verdicts),
        global_value=GlobalValue.R,
        topology=TopologyState(
            canonical=GeometricArrangement("expand_edge", (Perspective.OBJECTIVITY,)),
            surplus=GeometricArrangement("rest", ()),
            path=(),
        ),
        action=action,
    )


def result(*cycles: MetaCycle, selected: int = 0, reason: str = "converged") -> MetaResult:
    return MetaResult(
        question="why?",
        final_answer="because",
        selected_cycle=selected,
        termination_reason=reason,
        cycles=list(cycles),
    )


class TestRoundTrip:
    def test_a_saved_trace_loads_back(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(cycle(0, (True, True, False), "expand")), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert loaded.termination_reason == "converged"
        assert len(loaded.cycles) == 1

    def test_every_judgment_survives(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(cycle(0, (True, False, False), "refine")), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        judgments = loaded.cycles[0].judgments
        assert len(judgments) == 3
        assert {j.dimension for j in judgments} == set(Dimension)

    def test_rationales_survive(self, tmp_path: Path) -> None:
        """Attribution quotes them, so losing them makes the trace unreadable."""
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(cycle(0, (True, False, False), "refine")), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        objective = judgment_for(loaded.cycles[0], Dimension.OBJECTIVE)
        assert objective is not None
        assert objective.rationale == "o-rationale"

    def test_the_verdict_triple_survives(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(cycle(0, (True, True, False), "expand")), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert loaded.cycles[0].triple == "(A,W,R)"

    def test_several_cycles_keep_their_order(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(
            result(
                cycle(0, (True, True, False), "expand"),
                cycle(1, (True, False, False), "refine"),
                cycle(2, (True, True, True), "stop"),
            ),
            path,
        )

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert [c.index for c in loaded.cycles] == [0, 1, 2]
        assert [c.action for c in loaded.cycles] == ["expand", "refine", "stop"]

    def test_the_reinterpretation_flag_survives(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        reinterpreted = cycle(0, (True, True, True), "expand")
        flipped = MetaCycle(
            index=0,
            input_text=reinterpreted.input_text,
            all_inputs=reinterpreted.all_inputs,
            result=reinterpreted.result,
            evaluation=reinterpreted.evaluation.with_negative_dialectical("flipped"),
            global_value=GlobalValue.R,
            topology=reinterpreted.topology,
            action="expand",
        )
        save_meta_trace(result(flipped), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert loaded.cycles[0].reinterpreted

    def test_the_judging_method_survives(self, tmp_path: Path) -> None:
        """A fallback verdict must not read as a real judge verdict later."""
        path = tmp_path / "t.meta.json"
        fallback = PolyEvaluation(
            subjective=ContextJudgment.make(Dimension.SUBJECTIVE, True, "r", "fallback"),
            objective=ContextJudgment.make(Dimension.OBJECTIVE, True, "r"),
            dialectical=ContextJudgment.make(Dimension.DIALECTICAL, False, "r"),
        )
        base = cycle(0, (True, True, False), "expand")
        save_meta_trace(
            result(
                MetaCycle(
                    index=0,
                    input_text=base.input_text,
                    all_inputs=base.all_inputs,
                    result=base.result,
                    evaluation=fallback,
                    global_value=GlobalValue.R,
                    topology=base.topology,
                    action="expand",
                )
            ),
            path,
        )

        loaded = load_meta_trace(path)

        assert loaded is not None
        subjective = judgment_for(loaded.cycles[0], Dimension.SUBJECTIVE)
        assert subjective is not None
        assert subjective.method == "fallback"


class TestMissingAndPartial:
    def test_a_missing_file_loads_as_none(self, tmp_path: Path) -> None:
        """Summaries cached before traces were recorded have no trace beside
        them, which is not an error."""
        assert load_meta_trace(tmp_path / "absent.meta.json") is None

    def test_a_trace_with_no_cycles_loads(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(selected=-1, reason="no cycle completed"), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert loaded.cycles == ()

    def test_missing_optional_fields_default(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        path.write_text('{"cycles": [{"index": 0}]}', encoding="utf-8")

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert loaded.selected_cycle == -1
        assert loaded.cycles[0].judgments == ()

    def test_saving_creates_missing_directories(self, tmp_path: Path) -> None:
        path = tmp_path / "cache" / "runs" / "t.meta.json"
        save_meta_trace(result(cycle(0, (True, True, False), "expand")), path)

        assert path.is_file()


class TestSelection:
    def test_the_selected_cycle_is_returned(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(
            result(
                cycle(0, (True, True, False), "expand"),
                cycle(1, (True, True, True), "stop"),
                selected=1,
            ),
            path,
        )

        loaded = load_meta_trace(path)

        assert loaded is not None
        selected = loaded.selected()
        assert selected is not None and selected.index == 1

    def test_the_first_cycle_is_available_separately(self, tmp_path: Path) -> None:
        """Attribution compares the selected cycle against where it started."""
        path = tmp_path / "t.meta.json"
        save_meta_trace(
            result(
                cycle(0, (True, True, False), "expand"),
                cycle(1, (True, True, True), "stop"),
                selected=1,
            ),
            path,
        )

        loaded = load_meta_trace(path)

        assert loaded is not None
        first = loaded.first()
        assert first is not None and first.index == 0

    def test_no_cycles_means_no_selection(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(selected=-1), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        assert loaded.selected() is None
        assert loaded.first() is None

    def test_a_selection_pointing_at_no_cycle_falls_back_to_the_last(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        save_meta_trace(result(cycle(0, (True, True, False), "expand"), selected=7), path)

        loaded = load_meta_trace(path)

        assert loaded is not None
        selected = loaded.selected()
        assert selected is not None and selected.index == 0


class TestPathConvention:
    def test_the_trace_sits_beside_the_cached_summary(self) -> None:
        assert meta_trace_path(Path("results/cache/abc.txt")) == Path("results/cache/abc.meta.json")


class TestJudgmentLookup:
    def test_an_absent_dimension_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "t.meta.json"
        path.write_text('{"cycles": [{"index": 0}]}', encoding="utf-8")
        loaded = load_meta_trace(path)

        assert loaded is not None
        assert judgment_for(loaded.cycles[0], Dimension.OBJECTIVE) is None

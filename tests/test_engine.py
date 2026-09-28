"""The Sierpinski reflexion tree.

Driven by a counting stub client, so the assertions are about the structure the
engine builds and how many calls it takes to build it -- the two things that
decide what a run costs.
"""

from __future__ import annotations

import threading

import pytest

from polyrx.config import EngineConfig, ReflexionPrompts
from polyrx.engine import (
    Boundaries,
    Node,
    Perspective,
    PromptRegistry,
    ReflexionEngine,
    Region,
    StubLLMClient,
    extract_corner_answers,
    outermost_corner_leaf,
    reflexion,
)


def prompts() -> PromptRegistry:
    """A minimal template set. The engine only formats and dispatches them."""
    return PromptRegistry(
        ReflexionPrompts(
            boundaries={
                "objectivity_subjectivity": "boundary O-S of {input}",
                "objectivity_conceptuality": "boundary O-B of {input}",
                "conceptuality_subjectivity": "boundary B-S of {input}",
            },
            perspectives={
                "O": "objective view of {input} given {context}",
                "S": "subjective view of {input} given {context}",
                "B": "conceptual view of {input} given {context}",
            },
            summary="summarize {input}: {corner_o} / {corner_s} / {corner_b}",
            qa="answer {question} from {context} choosing {label_space}",
            judge="is {prediction} the same as {gold} among {label_space}",
        )
    )


class CountingClient:
    """Answers with the prompt it was given, and counts calls thread-safely."""

    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []
        self._lock = threading.Lock()

    def complete(self, prompt: str) -> str:
        with self._lock:
            self.calls += 1
            self.prompts.append(prompt)
        return f"answer to {prompt[:30]}"


class TestTreeShape:
    def test_depth_zero_is_a_single_leaf(self) -> None:
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=0) as engine:
            result = engine.run("a story")

        assert result.tree.is_leaf
        assert result.tree.perspectives is not None

    def test_every_node_branches_three_ways(self) -> None:
        """Three-way branching is the Sierpinski structure, not a setting."""
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=1) as engine:
            tree = engine.run("a story").tree

        assert tree.children is not None
        assert len(tree.children) == 3

    def test_leaves_sit_at_the_configured_depth(self) -> None:
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=2) as engine:
            tree = engine.run("a story").tree

        def depths(node: Node) -> list[int]:
            if node.is_leaf:
                return [node.depth]
            assert node.children is not None
            return [d for child in node.children for d in depths(child)]

        assert set(depths(tree)) == {2}

    def test_the_leaf_count_is_three_to_the_depth(self) -> None:
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=2) as engine:
            tree = engine.run("a story").tree

        def leaves(node: Node) -> int:
            if node.is_leaf:
                return 1
            assert node.children is not None
            return sum(leaves(child) for child in node.children)

        assert leaves(tree) == 9

    def test_each_child_gets_its_own_corner(self) -> None:
        """The loop-variable binding the engine comments on: without it every
        child would carry the last perspective."""
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=1) as engine:
            tree = engine.run("a story").tree

        assert tree.children is not None
        assert {child.region.apex for child in tree.children} == set(Perspective)


class TestCallCount:
    @pytest.mark.parametrize(
        ("depth", "expected"),
        [
            # leaves * 3 perspectives + internal nodes * 3 boundaries + 1 summary
            pytest.param(0, 3 + 0 + 1, id="depth-0"),
            pytest.param(1, 9 + 3 + 1, id="depth-1"),
            pytest.param(2, 27 + 12 + 1, id="depth-2"),
        ],
    )
    def test_calls_grow_with_the_tree(self, depth: int, expected: int) -> None:
        """What a condition costs. A change here changes every run's bill."""
        client = CountingClient()
        with ReflexionEngine(client, prompts=prompts(), max_depth=depth) as engine:
            engine.run("a story")

        assert client.calls == expected

    def test_serial_and_parallel_make_the_same_calls(self) -> None:
        serial, concurrent = CountingClient(), CountingClient()
        with ReflexionEngine(serial, prompts=prompts(), max_depth=2, parallel=False) as engine:
            engine.run("a story")
        with ReflexionEngine(concurrent, prompts=prompts(), max_depth=2) as engine:
            engine.run("a story")

        assert serial.calls == concurrent.calls


class TestResult:
    def test_one_answer_per_corner(self) -> None:
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=1) as engine:
            result = engine.run("a story")

        assert set(result.corner_answers) == set(Perspective)

    def test_the_summary_is_produced_from_the_corner_answers(self) -> None:
        client = CountingClient()
        with ReflexionEngine(client, prompts=prompts(), max_depth=0) as engine:
            result = engine.run("a story")

        assert result.summary
        assert any(p.startswith("summarize a story") for p in client.prompts)

    def test_a_plain_callable_is_accepted_as_a_client(self) -> None:
        """Notebooks and the interactive tools pass a function."""
        result = reflexion("a story", prompts(), max_depth=0, llm=lambda p: "fixed")

        assert result.summary == "fixed"

    def test_the_stub_client_drives_a_whole_run(self) -> None:
        """The offline path continuous integration relies on."""
        with ReflexionEngine(StubLLMClient(), prompts=prompts(), max_depth=1) as engine:
            result = engine.run("a story")

        assert result.summary
        assert len(result.corner_answers) == 3


class TestRegionsAndBoundaries:
    def test_the_root_region_has_no_boundaries(self) -> None:
        root = Region.root()

        assert root.left is None and root.right is None

    def test_an_edge_is_unordered(self) -> None:
        boundaries = Boundaries("O-S", "O-B", "B-S")

        assert boundaries.between(
            Perspective.OBJECTIVITY, Perspective.SUBJECTIVITY
        ) == boundaries.between(Perspective.SUBJECTIVITY, Perspective.OBJECTIVITY)

    def test_a_vertex_has_no_edge_to_itself(self) -> None:
        with pytest.raises(ValueError, match="No edge between"):
            Boundaries("a", "b", "c").between(Perspective.OBJECTIVITY, Perspective.OBJECTIVITY)

    def test_a_corner_takes_the_two_boundaries_adjacent_to_its_apex(self) -> None:
        boundaries = Boundaries("O-S", "O-B", "B-S")

        corner = Region.root().corner(Perspective.OBJECTIVITY, boundaries)

        assert corner.apex is Perspective.OBJECTIVITY
        assert {corner.left, corner.right} == {"O-S", "O-B"}


class TestCornerExtraction:
    def test_a_corner_follows_the_same_branch_at_every_level(self) -> None:
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=2) as engine:
            tree = engine.run("a story").tree

        leaf = outermost_corner_leaf(tree, Perspective.SUBJECTIVITY)

        assert leaf.is_leaf
        assert leaf.region.apex is Perspective.SUBJECTIVITY

    def test_extraction_takes_the_apex_matching_perspective(self) -> None:
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=1) as engine:
            tree = engine.run("a story").tree

        corners = extract_corner_answers(tree)
        leaf = outermost_corner_leaf(tree, Perspective.OBJECTIVITY)

        assert leaf.perspectives is not None
        assert corners[Perspective.OBJECTIVITY] == leaf.perspectives[Perspective.OBJECTIVITY]


class TestWorkerSizing:
    def test_the_pool_has_one_level_of_lookahead_beyond_the_deepest_level(self) -> None:
        """Too small and a depth-3 run deadlocks: parents wait on children that
        never get a thread."""
        assert ReflexionEngine.recommended_workers(2) == 27

    def test_the_ceiling_is_respected(self) -> None:
        config = EngineConfig(worker_pool_cap=10)

        assert ReflexionEngine.recommended_workers(5, config) == 10

    def test_a_deep_parallel_run_does_not_deadlock(self) -> None:
        """The reason the sizing rule exists."""
        with ReflexionEngine(CountingClient(), prompts=prompts(), max_depth=3) as engine:
            result = engine.run("a story")

        assert result.summary


class TestPromptRegistry:
    def test_a_prompt_set_without_a_scoring_template_says_so(self) -> None:
        """The interactive set has no question-answering pass."""
        registry = PromptRegistry(ReflexionPrompts(summary="s"))

        with pytest.raises(KeyError, match="no `qa` template"):
            registry.qa(context="c", question="q", label_space="Yes, No")

    def test_a_prompt_set_without_a_judge_template_says_so(self) -> None:
        registry = PromptRegistry(ReflexionPrompts(summary="s"))

        assert not registry.has_judge_template()
        with pytest.raises(KeyError, match="no `judge` template"):
            registry.judge(gold="Yes", prediction="No", label_space="Yes, No")

    def test_templates_are_formatted_with_their_fields(self) -> None:
        rendered = prompts().qa(context="the story", question="why?", label_space="Yes, No")

        assert "the story" in rendered and "why?" in rendered and "Yes, No" in rendered

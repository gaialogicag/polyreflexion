"""Recursive text reflexion over a Sierpinski triangle of three perspectives."""

from __future__ import annotations

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum

from polyrx.config import EngineConfig, OllamaConfig, ReflexionPrompts
from polyrx.models.base import CallableLLMClient, LLMClient
from polyrx.models.ollama import OllamaClient


class Perspective(Enum):
    """Vertices of the Sierpinski triangle."""

    OBJECTIVITY = "O"
    SUBJECTIVITY = "S"
    CONCEPTUALITY = "B"


_BOUNDARY_KEYS: dict[frozenset[Perspective], str] = {
    frozenset({Perspective.OBJECTIVITY, Perspective.SUBJECTIVITY}): "objectivity_subjectivity",
    frozenset({Perspective.OBJECTIVITY, Perspective.CONCEPTUALITY}): "objectivity_conceptuality",
    frozenset({Perspective.CONCEPTUALITY, Perspective.SUBJECTIVITY}): "conceptuality_subjectivity",
}

_BOUNDARY_EDGE_ORDER = (
    frozenset({Perspective.OBJECTIVITY, Perspective.SUBJECTIVITY}),
    frozenset({Perspective.OBJECTIVITY, Perspective.CONCEPTUALITY}),
    frozenset({Perspective.CONCEPTUALITY, Perspective.SUBJECTIVITY}),
)

_PERSPECTIVE_ORDER = (
    Perspective.OBJECTIVITY,
    Perspective.SUBJECTIVITY,
    Perspective.CONCEPTUALITY,
)


@dataclass(frozen=True)
class Boundaries:
    """Intersection points on the three edges (midpoints in the Sierpinski triangle)."""

    objectivity_subjectivity: str
    objectivity_conceptuality: str
    conceptuality_subjectivity: str

    def between(self, a: Perspective, b: Perspective) -> str:
        key = frozenset({a, b})
        match key:
            case s if s == frozenset({Perspective.OBJECTIVITY, Perspective.SUBJECTIVITY}):
                return self.objectivity_subjectivity
            case s if s == frozenset({Perspective.OBJECTIVITY, Perspective.CONCEPTUALITY}):
                return self.objectivity_conceptuality
            case s if s == frozenset({Perspective.CONCEPTUALITY, Perspective.SUBJECTIVITY}):
                return self.conceptuality_subjectivity
            case _:
                raise ValueError(f"No edge between {a} and {b}")


@dataclass(frozen=True)
class Region:
    """A triangle in the Sierpinski grid: apex plus two adjacent boundaries."""

    apex: Perspective
    left: str | None
    right: str | None

    @classmethod
    def root(cls) -> Region:
        """Root triangle with no boundaries yet."""
        return cls(apex=Perspective.OBJECTIVITY, left=None, right=None)

    def corner(self, perspective: Perspective, boundaries: Boundaries) -> Region:
        """Corner sub-triangle for one of the three remaining Sierpinski parts."""
        left_neighbor, right_neighbor = _neighbors(perspective)
        return Region(
            apex=perspective,
            left=boundaries.between(perspective, left_neighbor),
            right=boundaries.between(perspective, right_neighbor),
        )


def _neighbors(p: Perspective) -> tuple[Perspective, Perspective]:
    """Both neighboring vertices (counter-clockwise from the apex)."""
    i = _PERSPECTIVE_ORDER.index(p)
    return _PERSPECTIVE_ORDER[(i + 1) % 3], _PERSPECTIVE_ORDER[(i + 2) % 3]


@dataclass
class Node:
    """Result of one recursion step: either a leaf or a branching node."""

    region: Region
    depth: int
    boundaries: Boundaries | None = None
    perspectives: dict[Perspective, str] | None = None
    children: tuple[Node, Node, Node] | None = None

    @property
    def is_leaf(self) -> bool:
        return self.children is None


@dataclass
class ReflexionResult:
    """Full reflexion output: tree, outermost corner answers, and synthesis."""

    tree: Node
    corner_answers: dict[Perspective, str]
    summary: str


class StubLLMClient:
    """Deterministic stub for tests without a real LLM."""

    def complete(self, prompt: str) -> str:
        if "Write the single best possible answer" in prompt:
            return "[Best answer to the question]"
        if "Synthesize these three corner perspectives" in prompt:
            return "[Integrated summary of O, S, and B corner perspectives]"
        if "integrated summary of the story" in prompt:
            return "[Integrated summary of O, S, and B corner perspectives]"
        if "integrated scientific reasoning summary" in prompt:
            return "[Integrated scientific reasoning summary of O, S, and B]"
        if "boundary between" in prompt:
            boundary = prompt.split("boundary between")[1].split("?")[0].strip()
            return f"[Boundary: {boundary}]"
        if "objective perspective" in prompt or "objective standpoint" in prompt:
            return "[Objective rewrite]"
        if "subjective perspective" in prompt or "subjective standpoint" in prompt:
            return "[Subjective rewrite]"
        if "conceptual perspective" in prompt or "conceptual standpoint" in prompt:
            return "[Conceptual rewrite]"
        if "Allowed answers" in prompt and "Gold answer:" not in prompt:
            # OpenToM: "choose exactly one):"; GPQA: "choose exactly one letter):"
            marker = (
                "Allowed answers (choose exactly one letter):"
                if "choose exactly one letter" in prompt
                else "Allowed answers (choose exactly one):"
            )
            for part in prompt.split(marker)[-1].split("\n")[0].split(","):
                label = part.strip()
                if label:
                    return label
        if "Gold answer:" in prompt:
            gold = prompt.split("Gold answer:")[1].split("\n")[0].strip()
            pred = prompt.split("Model answer:")[1].split("\nAllowed")[0].strip()
            from polyrx.benchmark.judge import labels_match

            correct = labels_match(pred, gold)
            return json.dumps({"correct": correct, "normalized_answer": pred})
        return "[LLM response]"


class PromptRegistry:
    """Formats the engine's templates.

    Built from a :class:`~polyrx.config.ReflexionPrompts`, which Hydra
    composes from ``conf/prompts/reflexion/``. Nothing here reads a file: the
    templates are configuration, so they are selected, overridden and recorded
    the same way every other setting is.
    """

    def __init__(self, prompts: ReflexionPrompts) -> None:
        self._boundaries = dict(prompts.boundaries)
        self._perspectives = {Perspective(k): v for k, v in prompts.perspectives.items()}
        self._summary = prompts.summary
        # Empty in the interactive set, which has no scoring pass.
        self._qa = prompts.qa
        self._judge = prompts.judge

    def boundary(self, edge: frozenset[Perspective]) -> str:
        return self._boundaries[_BOUNDARY_KEYS[edge]]

    def perspective(self, p: Perspective) -> str:
        return self._perspectives[p]

    def summary(
        self,
        *,
        input_text: str,
        corner_o: str,
        corner_s: str,
        corner_b: str,
    ) -> str:
        return self._summary.format(
            input=input_text,
            corner_o=corner_o,
            corner_s=corner_s,
            corner_b=corner_b,
        )

    def has_judge_template(self) -> bool:
        """Whether a judge template is configured for this prompt set."""
        return bool(self._judge)

    def qa(self, *, context: str, question: str, label_space: str) -> str:
        """Ask the question, given whatever context the condition produced."""
        if not self._qa:
            raise KeyError(
                "This prompt set has no `qa` template, so it cannot score a "
                "benchmark. Select one that does, e.g. prompts/reflexion=default."
            )
        return self._qa.format(context=context, question=question, label_space=label_space)

    def judge(self, *, gold: str, prediction: str, label_space: str) -> str:
        """Ask a model whether an answer matches the gold label."""
        if not self._judge:
            raise KeyError("This prompt set has no `judge` template.")
        return self._judge.format(gold=gold, prediction=prediction, label_space=label_space)


class ReflexionEngine:
    """Parallel Sierpinski reflexion orchestrator.

    Parallelizes at three levels:
    - boundary queries within a node (3 concurrent LLM calls)
    - perspective resolution at leaves (3 concurrent LLM calls)
    - sibling subtrees (3 concurrent recursion branches)

    Uses two thread pools to avoid deadlocks when nested tasks wait on children.
    """

    def __init__(
        self,
        llm: LLMClient | Callable[[str], str],
        *,
        prompts: PromptRegistry,
        max_depth: int | None = None,
        max_workers: int | None = None,
        parallel: bool = True,
        config: EngineConfig | None = None,
    ) -> None:
        self._llm = llm if isinstance(llm, LLMClient) else CallableLLMClient(llm)
        self._prompts = prompts
        self._config = config or EngineConfig()
        max_depth = self._config.default_max_depth if max_depth is None else max_depth
        self._max_depth = max_depth
        self._parallel = parallel
        self._llm_pool: ThreadPoolExecutor | None = None
        self._tree_pool: ThreadPoolExecutor | None = None
        if parallel:
            # Tree recursion submits child tasks and waits on the same pool.
            # That pool MUST be wide enough for the full breadth of waiting
            # parents + queued children, or depth>=3 deadlocks.
            tree_workers = self.recommended_workers(max_depth, self._config)
            # Cap LLM concurrency separately (API rate limits / local GPU).
            llm_workers = max_workers or tree_workers
            self._llm_pool = ThreadPoolExecutor(max_workers=llm_workers, thread_name_prefix="llm")
            self._tree_pool = ThreadPoolExecutor(
                max_workers=tree_workers, thread_name_prefix="tree"
            )

    @staticmethod
    def recommended_workers(max_depth: int, config: EngineConfig | None = None) -> int:
        """Worker count for full tree + batched LLM parallelism without deadlock.

        The tree branches three ways per node — that is the Sierpinski
        structure, not a setting — so the pool needs one level of lookahead
        beyond the deepest level. Only the ceiling is configurable.
        """
        config = config or EngineConfig()
        return min(config.worker_pool_cap, len(Perspective) ** (max_depth + 1))

    def run(self, text: str) -> ReflexionResult:
        """Run parallel reflexion and synthesize outermost corner perspectives."""
        tree = self._reflexion(text, Region.root(), depth=0)
        corner_answers = extract_corner_answers(tree)
        summary = self._summarize(text, corner_answers)
        return ReflexionResult(tree=tree, corner_answers=corner_answers, summary=summary)

    def _summarize(self, text: str, corners: dict[Perspective, str]) -> str:
        prompt = self._prompts.summary(
            input_text=text,
            corner_o=corners[Perspective.OBJECTIVITY],
            corner_s=corners[Perspective.SUBJECTIVITY],
            corner_b=corners[Perspective.CONCEPTUALITY],
        )
        return self._llm.complete(prompt)

    def close(self) -> None:
        if self._llm_pool is not None:
            self._llm_pool.shutdown(wait=True)
        if self._tree_pool is not None:
            self._tree_pool.shutdown(wait=True)

    def __enter__(self) -> ReflexionEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _parallel_llm(self, prompts: list[str]) -> list[str]:
        if not self._parallel or self._llm_pool is None:
            return [self._llm.complete(p) for p in prompts]
        futures = [self._llm_pool.submit(self._llm.complete, p) for p in prompts]
        return [f.result() for f in futures]

    def _parallel_tree(self, tasks: list[Callable[[], Node]]) -> tuple[Node, ...]:
        if not self._parallel or self._tree_pool is None:
            return tuple(task() for task in tasks)
        futures = [self._tree_pool.submit(task) for task in tasks]
        return tuple(f.result() for f in futures)

    def _enriched_input(self, text: str, region: Region) -> str:
        return f"{text}\n\n[Region]\n{self._format_context(region)}"

    def _find_boundaries(self, text: str, region: Region) -> Boundaries:
        enriched = self._enriched_input(text, region)
        prompts = [
            self._prompts.boundary(edge).format(input=enriched) for edge in _BOUNDARY_EDGE_ORDER
        ]
        os_, ob, bs = self._parallel_llm(prompts)
        return Boundaries(
            objectivity_subjectivity=os_,
            objectivity_conceptuality=ob,
            conceptuality_subjectivity=bs,
        )

    def _resolve_perspectives(self, text: str, region: Region) -> dict[Perspective, str]:
        context = self._format_context(region)
        prompts = [
            self._prompts.perspective(p).format(input=text, context=context)
            for p in _PERSPECTIVE_ORDER
        ]
        answers = self._parallel_llm(prompts)
        return dict(zip(_PERSPECTIVE_ORDER, answers, strict=True))

    def _reflexion(self, text: str, region: Region, depth: int) -> Node:
        if depth >= self._max_depth:
            return Node(
                region=region,
                depth=depth,
                perspectives=self._resolve_perspectives(text, region),
            )

        boundaries = self._find_boundaries(text, region)
        children = self._parallel_tree(
            [
                # `p=p` binds the loop variable at definition time, so each
                # lambda captures its own perspective rather than the last one.
                # mypy cannot infer a lambda with a bound default here.
                lambda p=p: self._reflexion(  # type: ignore[misc]
                    text,
                    region.corner(p, boundaries),
                    depth + 1,
                )
                for p in _PERSPECTIVE_ORDER
            ]
        )

        return Node(
            region=region,
            depth=depth,
            boundaries=boundaries,
            children=children,  # type: ignore[arg-type]
        )

    @staticmethod
    def _format_context(region: Region) -> str:
        parts = [f"Apex: {region.apex.value}"]
        if region.left is not None:
            parts.append(f"Left boundary: {region.left}")
        if region.right is not None:
            parts.append(f"Right boundary: {region.right}")
        return "\n".join(parts)


def call_llm(prompt: str, config: OllamaConfig | None = None) -> str:
    """One-shot Ollama call returning only the final answer.

    Convenience for scripts and notebooks; the benchmarks go through
    :func:`polyrx.models.registry.get_client` instead.
    """
    return OllamaClient(config).complete(prompt)


def outermost_corner_leaf(tree: Node, corner: Perspective) -> Node:
    """Follow the same corner branch at every level to the outermost leaf."""
    idx = _PERSPECTIVE_ORDER.index(corner)
    node = tree
    while not node.is_leaf:
        assert node.children is not None
        node = node.children[idx]
    return node


def extract_corner_answers(tree: Node) -> dict[Perspective, str]:
    """Collect the apex-matching perspective at each outermost triangle corner."""
    corners: dict[Perspective, str] = {}
    for corner in _PERSPECTIVE_ORDER:
        leaf = outermost_corner_leaf(tree, corner)
        assert leaf.perspectives is not None
        corners[corner] = leaf.perspectives[corner]
    return corners


def reflexion(
    text: str,
    prompts: PromptRegistry,
    *,
    max_depth: int | None = None,
    llm: LLMClient | Callable[[str], str] = call_llm,
    max_workers: int | None = None,
    config: EngineConfig | None = None,
) -> ReflexionResult:
    """Run parallel reflexion (convenience wrapper around ReflexionEngine).

    ``prompts`` is required: templates are configuration, so there is no
    implicit default set to fall back on.
    """
    with ReflexionEngine(
        llm,
        prompts=prompts,
        max_depth=max_depth,
        max_workers=max_workers,
        config=config,
    ) as engine:
        return engine.run(text)


def _stub_llm(prompt: str) -> str:
    """Minimal stub for local tests without a real LLM."""
    return StubLLMClient().complete(prompt)


def print_tree(node: Node, indent: int = 0) -> None:
    """Print the reflexion result tree."""
    prefix = "  " * indent
    apex = node.region.apex.value
    print(f"{prefix}▲ {apex} (depth {node.depth})")

    if node.is_leaf:
        for p in Perspective:
            assert node.perspectives is not None
            print(f"{prefix}  {p.value}: {node.perspectives[p]}")
        return

    assert node.boundaries is not None and node.children is not None
    print(f"{prefix}  boundaries: O↔S, O↔B, B↔S")
    for child in node.children:
        print_tree(child, indent + 1)


_CORNER_LABELS = {
    Perspective.OBJECTIVITY: "Objectivity (O)",
    Perspective.SUBJECTIVITY: "Subjectivity (S)",
    Perspective.CONCEPTUALITY: "Conceptuality (B)",
}


def print_final(result: ReflexionResult) -> None:
    """Print outermost corner answers and the integrated summary."""
    print("\n" + "=" * 60)
    print("OUTERMOST CORNERS")
    print("=" * 60)
    for corner in _PERSPECTIVE_ORDER:
        print(f"\n{_CORNER_LABELS[corner]}:")
        print(result.corner_answers[corner])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{result.summary}")

"""Core data structures for the polycontextural meta-evaluation layer.

The meta layer treats one completed ``ReflexionEngine`` run as a single
observable object.  These types describe what it observes (``Observation``),
what it concludes (``PolyEvaluation`` -> ``GlobalValue``), where that places
the reasoning on the Sierpinski triangle (``TopologyState``), and what it
organizes next (``Decision`` / ``MetaCycle`` / ``MetaResult``).

Nothing in this module talks to an LLM; it is pure structure.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum

from polyreflexion.engine import Perspective, ReflexionResult


class Dimension(Enum):
    """The three polycontextural evaluation contexts."""

    SUBJECTIVE = "subjective"
    OBJECTIVE = "objective"
    DIALECTICAL = "dialectical"


# Canonical iteration order (stable everywhere: judging, ties, display).
DIMENSION_ORDER: tuple[Dimension, ...] = (
    Dimension.SUBJECTIVE,
    Dimension.OBJECTIVE,
    Dimension.DIALECTICAL,
)

# Each context is anchored at one vertex of the existing reflexion triangle,
# so evaluation results translate directly into triangle geometry.
DIMENSION_TO_VERTEX: dict[Dimension, Perspective] = {
    Dimension.SUBJECTIVE: Perspective.SUBJECTIVITY,
    Dimension.OBJECTIVE: Perspective.OBJECTIVITY,
    Dimension.DIALECTICAL: Perspective.CONCEPTUALITY,
}

# Display labels for the positive / negative pole of each context.
POSITIVE_LABEL: dict[Dimension, str] = {
    Dimension.SUBJECTIVE: "A",   # authentic
    Dimension.OBJECTIVE: "W",    # true
    Dimension.DIALECTICAL: "W",  # productive
}
NEGATIVE_LABEL: dict[Dimension, str] = {
    Dimension.SUBJECTIVE: "F",   # inauthentic
    Dimension.OBJECTIVE: "F",    # false
    Dimension.DIALECTICAL: "R",  # redundant
}


def summary_hash(text: str) -> str:
    """Stable content hash for oscillation detection and trace records."""
    normalized = " ".join(text.split()).lower()
    return hashlib.sha1(normalized.encode()).hexdigest()


@dataclass(frozen=True)
class ContextJudgment:
    """Verdict of one judge in one logical context."""

    dimension: Dimension
    positive: bool
    label: str      # display label: A/F (subjective), W/F (objective), W/R (dialectical)
    rationale: str  # must name the statements that drove the verdict
    method: str     # "llm_judge" | "fallback" | "reinterpretation"

    @classmethod
    def make(
        cls,
        dimension: Dimension,
        positive: bool,
        rationale: str,
        method: str = "llm_judge",
    ) -> ContextJudgment:
        """Build a judgment with the display label derived from the pole."""
        label = POSITIVE_LABEL[dimension] if positive else NEGATIVE_LABEL[dimension]
        return cls(
            dimension=dimension,
            positive=positive,
            label=label,
            rationale=rationale,
            method=method,
        )

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "positive": self.positive,
            "label": self.label,
            "rationale": self.rationale,
            "method": self.method,
        }


@dataclass(frozen=True)
class PolyEvaluation:
    """The complete observable valuation triple (subjective, objective, dialectical)."""

    subjective: ContextJudgment
    objective: ContextJudgment
    dialectical: ContextJudgment
    reinterpreted: bool = False  # True after the (A,W,W) -> (A,W,R) rule fired

    def judgments(self) -> tuple[ContextJudgment, ContextJudgment, ContextJudgment]:
        return (self.subjective, self.objective, self.dialectical)

    def positives(self) -> tuple[Dimension, ...]:
        """Dimensions judged positively, in canonical order."""
        return tuple(j.dimension for j in self.judgments() if j.positive)

    def negatives(self) -> tuple[ContextJudgment, ...]:
        """Negative judgments, in canonical order (their rationales become boundaries)."""
        return tuple(j for j in self.judgments() if not j.positive)

    def fallback_count(self) -> int:
        return sum(1 for j in self.judgments() if j.method == "fallback")

    def as_triple(self) -> str:
        """Display form, e.g. ``(F,W,W)``."""
        return "(" + ",".join(j.label for j in self.judgments()) + ")"

    def feedback_block(self) -> str:
        """All three judge verdicts with rationales for the next engine input."""
        lines: list[str] = []
        for judgment in self.judgments():
            if judgment.positive:
                status = "satisfied — preserve this strength"
            else:
                status = "NOT satisfied — the next answer must fix this"
            lines.append(
                f"- {judgment.dimension.value} ({judgment.label}, {status}): "
                f"{judgment.rationale}"
            )
        return "\n".join(lines)

    def repair_priorities(self) -> str:
        """Rationales from failing judges only — what the next answer must address."""
        negatives = self.negatives()
        if not negatives:
            return "(All judges passed — extend what already works without redundancy.)"
        return "\n".join(
            f"- {judgment.dimension.value} ({judgment.label}): {judgment.rationale}"
            for judgment in negatives
        )

    def guidance_for(self, dimension: Dimension) -> str:
        """The rationale for one dimension's verdict (used in boundary generation)."""
        for judgment in self.judgments():
            if judgment.dimension is dimension:
                pole = "positive" if judgment.positive else "negative"
                return f"{judgment.label} ({pole}): {judgment.rationale}"
        raise ValueError(f"No judgment for {dimension!r}")

    def with_negative_dialectical(
        self,
        rationale: str,
        method: str = "reinterpretation",
    ) -> PolyEvaluation:
        """Return a copy with the dialectical verdict flipped to negative.

        Used by the (A,W,W) -> (A,W,R) anti-convergence rule and by the
        low-confidence guard.
        """
        return PolyEvaluation(
            subjective=self.subjective,
            objective=self.objective,
            dialectical=ContextJudgment.make(
                Dimension.DIALECTICAL, False, rationale, method
            ),
            reinterpreted=True,
        )

    def to_dict(self) -> dict:
        return {
            "triple": self.as_triple(),
            "reinterpreted": self.reinterpreted,
            "judgments": [j.to_dict() for j in self.judgments()],
        }


class GlobalValue(Enum):
    """Overall interpretation of a valuation triple."""

    W = "W"  # all three positive: converged
    F = "F"  # none positive: reset
    R = "R"  # two positive: boundary of the semantic space -> expansion
    A = "A"  # one positive: enough negative structure -> refinement


@dataclass(frozen=True)
class GeometricArrangement:
    """One concrete placement/move in the Sierpinski triangle."""

    kind: str  # expand_edge | descend_corner | rest | reset_root | expand_full | invert_root
    anchor: tuple[Perspective, ...]  # edge = 2 vertices, corner = 1, global moves = 0 or 3

    def to_dict(self) -> dict:
        return {"kind": self.kind, "anchor": [p.value for p in self.anchor]}


@dataclass(frozen=True)
class TopologyState:
    """Canonical move plus the preserved unused alternative (semantic surplus)."""

    canonical: GeometricArrangement
    surplus: GeometricArrangement  # never drives control flow in v1; kept for future meta-levels
    path: tuple[str, ...] = ()     # moves from the root, e.g. ("S", "+OS")

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical.to_dict(),
            "surplus": self.surplus.to_dict(),
            "path": list(self.path),
        }


@dataclass(frozen=True)
class BoundaryStatement:
    """A semantic boundary used as a coordinate for the next recursive cycle."""

    dimension: Dimension
    origin: str            # "negation_of_positive" (expansion) | "reuse_of_negative" (refinement)
    text: str              # the boundary description itself
    source_rationale: str  # judge rationale it was derived from

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "origin": self.origin,
            "text": self.text,
            "source_rationale": self.source_rationale,
        }


@dataclass(frozen=True)
class Observation:
    """Everything the judges may look at for one completed engine run."""

    input_text: str                            # the original question / text
    summary: str                               # the engine's final answer
    corner_answers: dict[Perspective, str]     # O/S/B reasoning artifacts
    previous_summary: str | None = None        # prior cycle's answer (dialectical context)

    def corners_block(self) -> str:
        """Render the corner artifacts as a compact prompt block."""
        return "\n".join(f"{p.value}: {text}" for p, text in self.corner_answers.items())


@dataclass
class Decision:
    """What the strategy planner tells the controller to do next."""

    action: str  # "expand" | "refine" | "reset" | "terminate"
    next_inputs: list[str] = field(default_factory=list)
    engine_depth: int = 1
    boundaries: list[BoundaryStatement] = field(default_factory=list)
    reason: str = ""


@dataclass
class MetaCycle:
    """One full observe -> evaluate -> decide record."""

    index: int
    input_text: str                # input of the chosen candidate
    all_inputs: tuple[str, ...]    # every input executed this cycle (expansion may run 2)
    result: ReflexionResult        # chosen engine result (selection, never editing)
    evaluation: PolyEvaluation
    global_value: GlobalValue
    topology: TopologyState
    boundaries: list[BoundaryStatement] = field(default_factory=list)
    action: str = ""
    next_inputs: list[str] = field(default_factory=list)
    drifted: bool = False
    candidate_summaries: list[str] = field(default_factory=list)

    def positives_count(self) -> int:
        return len(self.evaluation.positives())

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "input_text": self.input_text,
            "all_inputs": list(self.all_inputs),
            "summary": self.result.summary,
            "summary_sha1": summary_hash(self.result.summary),
            "corner_answers": {p.value: t for p, t in self.result.corner_answers.items()},
            "evaluation": self.evaluation.to_dict(),
            "global_value": self.global_value.value,
            "topology": self.topology.to_dict(),
            "boundaries": [b.to_dict() for b in self.boundaries],
            "action": self.action,
            "next_inputs": self.next_inputs,
            "drifted": self.drifted,
            "candidate_summaries": self.candidate_summaries,
        }


@dataclass
class MetaResult:
    """Final outcome of a meta run: a selected answer plus the full cycle history."""

    question: str
    final_answer: str
    selected_cycle: int  # -1 when no cycle completed
    termination_reason: str
    cycles: list[MetaCycle] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "final_answer": self.final_answer,
            "selected_cycle": self.selected_cycle,
            "termination_reason": self.termination_reason,
            "cycles": [c.to_dict() for c in self.cycles],
        }


@dataclass(frozen=True)
class MetaConfig:
    """Tunable knobs of the meta loop (all guards live here)."""

    max_cycles: int = 4                 # hard recursion budget
    engine_depth: int = 1               # reflexion depth per cycle
    reset_depth_bonus: int = 1          # extra depth for the one F-reset re-run
    reinterpret_full_positive_on_cycle0: bool = True  # cycle-0 (A,W,W)->(A,W,R) only
    drift_check_enabled: bool = True
    max_consecutive_drifts: int = 2

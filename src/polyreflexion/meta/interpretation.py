"""Deterministic interpretation of a valuation triple into a global value.

The rule is a pure lookup on the number of positive verdicts:

    3 positives -> W (converged)
    2 positives -> R (boundary reached -> expansion)
    1 positive  -> A (enough negative structure -> refinement)
    0 positives -> F (reset)

plus the anti-convergence exception on the **first meta cycle only** (cycle 0):

When meta first observes the initial reflexion and judges return (A,W,W), the
dialectical verdict is flipped to R so the loop expands instead of stopping on
what is likely dialectical exhaustion.  On every later meta cycle, a genuine
(A,W,W) is left intact and maps to GlobalValue W (converged / stop).
"""

from __future__ import annotations

from polyreflexion.meta.datatypes import GlobalValue, PolyEvaluation

# Default lookup; kept as data so future meta-levels can override it.
DEFAULT_LOOKUP: dict[int, GlobalValue] = {
    3: GlobalValue.W,
    2: GlobalValue.R,
    1: GlobalValue.A,
    0: GlobalValue.F,
}

_REINTERPRETATION_RATIONALE = (
    "Reinterpreted (A,W,W) -> (A,W,R) on meta cycle 0 only: the initial "
    "reflexion looked fully positive but is assumed dialectically exhausted; "
    "meta continues with boundary expansion."
)


class Interpreter:
    """Map a PolyEvaluation to a GlobalValue via a configurable lookup."""

    def __init__(
        self,
        lookup: dict[int, GlobalValue] | None = None,
        *,
        reinterpret_full_positive_on_cycle0: bool = True,
    ) -> None:
        self._lookup = dict(lookup or DEFAULT_LOOKUP)
        self._reinterpret = reinterpret_full_positive_on_cycle0

    def interpret(
        self,
        evaluation: PolyEvaluation,
        cycle_index: int,
    ) -> tuple[GlobalValue, PolyEvaluation]:
        """Return (global value, possibly reinterpreted evaluation).

        (A,W,W) handling — only on meta cycle 0 (first observation of the
        initial reflexion, before meta has steered any expansion/refinement):

            judges say (A,W,W)  ->  flip dialectical to R, continue as expansion

        On meta cycle 1 and later, (A,W,W) is never reinterpreted; three
        positives map straight to GlobalValue W and the controller stops.
        """
        if self._reinterpret and cycle_index == 0 and len(evaluation.positives()) == 3:
            evaluation = evaluation.with_negative_dialectical(_REINTERPRETATION_RATIONALE)

        return self._lookup[len(evaluation.positives())], evaluation

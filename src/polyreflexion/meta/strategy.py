"""Strategy planner: the action side of the meta state machine.

Given a global value, the planner composes the next engine inputs:

- **W** -> terminate (converged).
- **R** -> expansion: negation boundaries of the positive dimensions become up
  to two new engine inputs ("construct the missing semantic coordinate").
- **A** -> refinement: the two negative rationales become the left/right
  coordinates of a sub-triangle input; the positive dimension is the apex.
- **F** -> one reset re-run of the original input at increased depth; a second
  F terminates with ``exhausted_F``.
"""

from __future__ import annotations

from polyreflexion.meta.boundaries import BoundaryGenerator
from polyreflexion.meta.datatypes import (
    DIMENSION_TO_VERTEX,
    Decision,
    GlobalValue,
    MetaConfig,
    PolyEvaluation,
)
from polyreflexion.meta.prompts import MetaPromptRegistry


class StrategyPlanner:
    """Turn (GlobalValue, PolyEvaluation) into a concrete Decision."""

    def __init__(
        self,
        prompts: MetaPromptRegistry,
        boundaries: BoundaryGenerator,
        config: MetaConfig,
    ) -> None:
        self._prompts = prompts
        self._boundaries = boundaries
        self._config = config

    def decide(
        self,
        *,
        global_value: GlobalValue,
        evaluation: PolyEvaluation,
        question: str,
        answer: str,
        reset_used: bool,
    ) -> Decision:
        if global_value is GlobalValue.W:
            return Decision(action="terminate", reason="converged_W")

        if global_value is GlobalValue.R:
            return self._expand(evaluation, question, answer)

        if global_value is GlobalValue.A:
            return self._refine(evaluation, question, answer)

        # GlobalValue.F — undefined by the spec; plan decision: reset once.
        if not reset_used:
            return Decision(
                action="reset",
                next_inputs=[question],
                engine_depth=self._config.engine_depth + self._config.reset_depth_bonus,
                reason="reset_after_F",
            )
        return Decision(action="terminate", reason="exhausted_F")

    def _expand(self, evaluation: PolyEvaluation, question: str, answer: str) -> Decision:
        """Expansion: one new input per (non-contradictory) negation boundary."""
        statements, dropped = self._boundaries.negations(
            evaluation, answer, question=question
        )
        feedback = evaluation.feedback_block()
        inputs = [
            self._prompts.expand_input(
                question=question,
                answer=answer,
                boundary=statement.text,
                dimension=statement.dimension.value,
                judge_feedback=feedback,
            )
            for statement in statements
        ]
        reason = "expansion"
        if dropped:
            reason += " (contradictory boundary dropped)"
        return Decision(
            action="expand",
            next_inputs=inputs,
            engine_depth=self._config.engine_depth,
            boundaries=statements,
            reason=reason,
        )

    def _refine(self, evaluation: PolyEvaluation, question: str, answer: str) -> Decision:
        """Refinement: descend into the sub-triangle bounded by the negatives."""
        boundaries = self._boundaries.reuse_negatives(evaluation)
        (positive,) = evaluation.positives()
        left, right = boundaries[0], boundaries[1]
        refined_input = self._prompts.refine_input(
            question=question,
            answer=answer,
            apex=DIMENSION_TO_VERTEX[positive].value,
            apex_guidance=evaluation.guidance_for(positive),
            left_label=left.dimension.value,
            left_boundary=left.text,
            right_label=right.dimension.value,
            right_boundary=right.text,
            judge_feedback=evaluation.feedback_block(),
        )
        return Decision(
            action="refine",
            next_inputs=[refined_input],
            engine_depth=self._config.engine_depth,
            boundaries=boundaries,
            reason="refinement",
        )

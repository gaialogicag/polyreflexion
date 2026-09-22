"""Boundary statement construction.

Two origins, per the spec:

- **Expansion (R)**: for each positively judged dimension, an LLM call derives
  the *negation boundary* — what still separates the answer from its opposite.
  The two calls run in parallel; a cheap contradiction check guards the pair.
- **Refinement (A)**: no generation at all — the two negative judges'
  rationales are reused verbatim as the sub-triangle boundaries.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from polyreflexion.meta.datatypes import BoundaryStatement, ContextJudgment, PolyEvaluation
from polyreflexion.meta.prompts import MetaPromptRegistry


class BoundaryGenerator:
    """Produce boundary statements from a poly-evaluation."""

    def __init__(
        self,
        client,  # LLMClient-compatible: .complete(prompt) -> str
        prompts: MetaPromptRegistry,
        *,
        parallel: bool = True,
    ) -> None:
        self._client = client
        self._prompts = prompts
        self._parallel = parallel

    def negations(
        self,
        evaluation: PolyEvaluation,
        answer: str,
        *,
        question: str = "",
    ) -> tuple[list[BoundaryStatement], bool]:
        """Negation boundaries for every positive dimension (expansion).

        Returns ``(statements, contradiction_dropped)``.  When the two
        boundaries contradict each other, the one from the earlier dimension in
        canonical order is kept (deterministic tie-break) and the flag is set
        so the controller can log the conflict.

        Each boundary is generated with the full judge triple so failing
        verdicts (especially dialectical redundancy) shape the next cycle.
        """
        positive_judgments = [j for j in evaluation.judgments() if j.positive]
        judge_feedback = evaluation.feedback_block()
        repair_priorities = evaluation.repair_priorities()

        if self._parallel and len(positive_judgments) > 1:
            with ThreadPoolExecutor(
                max_workers=len(positive_judgments), thread_name_prefix="boundary"
            ) as pool:
                statements = list(
                    pool.map(
                        lambda j: self._negate_one(
                            j,
                            answer,
                            question=question,
                            judge_feedback=judge_feedback,
                            repair_priorities=repair_priorities,
                        ),
                        positive_judgments,
                    )
                )
        else:
            statements = [
                self._negate_one(
                    j,
                    answer,
                    question=question,
                    judge_feedback=judge_feedback,
                    repair_priorities=repair_priorities,
                )
                for j in positive_judgments
            ]

        dropped = False
        if len(statements) == 2 and self._contradictory(statements[0].text, statements[1].text):
            statements = [statements[0]]
            dropped = True
        return statements, dropped

    def reuse_negatives(self, evaluation: PolyEvaluation) -> list[BoundaryStatement]:
        """Refinement boundaries: negative rationales reused verbatim (no LLM)."""
        return [
            BoundaryStatement(
                dimension=judgment.dimension,
                origin="reuse_of_negative",
                text=judgment.rationale,
                source_rationale=judgment.rationale,
            )
            for judgment in evaluation.negatives()
        ]

    def _negate_one(
        self,
        judgment: ContextJudgment,
        answer: str,
        *,
        question: str,
        judge_feedback: str,
        repair_priorities: str,
    ) -> BoundaryStatement:
        prompt = self._prompts.boundary_negation(
            dimension=judgment.dimension,
            question=question,
            answer=answer,
            rationale=judgment.rationale,
            verdict_label=judgment.label,
            judge_feedback=judge_feedback,
            repair_priorities=repair_priorities,
        )
        text = self._client.complete(prompt).strip()
        return BoundaryStatement(
            dimension=judgment.dimension,
            origin="negation_of_positive",
            text=text,
            source_rationale=judgment.rationale,
        )

    def _contradictory(self, first: str, second: str) -> bool:
        prompt = self._prompts.contradiction_check(first=first, second=second)
        try:
            reply = self._client.complete(prompt).strip().lower()
        except Exception:
            return False  # fail open: a broken guard must not drop boundaries
        return reply.startswith("y")

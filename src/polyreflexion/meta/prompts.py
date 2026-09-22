"""Prompt registry for the meta layer.

Kept separate from the engine's ``PromptRegistry`` so meta templates can evolve
without touching the proven reflexion ones. The sets live in
``conf/prompts/meta/``; a literal brace inside a template is written ``{{``.
"""

from __future__ import annotations

from dataclasses import asdict

from polyreflexion.config import MetaPrompts
from polyreflexion.meta.datatypes import Dimension

_JUDGE_KEYS: dict[Dimension, str] = {
    Dimension.SUBJECTIVE: "judge_subjective",
    Dimension.OBJECTIVE: "judge_objective",
    Dimension.DIALECTICAL: "judge_dialectical",
}


class MetaPromptRegistry:
    """Formats the meta-layer templates from a configured set."""

    def __init__(self, prompts: MetaPrompts) -> None:
        self._templates: dict[str, str] = asdict(prompts)

    def _format(self, key: str, **kwargs: str) -> str:
        return self._templates[key].format(**kwargs)

    def judge(
        self,
        dimension: Dimension,
        *,
        question: str,
        answer: str,
        corners: str,
        previous_answer: str,
    ) -> str:
        # str.format ignores unused keyword arguments, so every judge template
        # can pick the subset of fields it actually needs.
        return self._format(
            _JUDGE_KEYS[dimension],
            question=question,
            answer=answer,
            corners=corners,
            previous_answer=previous_answer,
        )

    def boundary_negation(
        self,
        *,
        dimension: Dimension,
        question: str,
        answer: str,
        rationale: str,
        verdict_label: str,
        judge_feedback: str,
        repair_priorities: str,
    ) -> str:
        return self._format(
            "boundary_negation",
            dimension=dimension.value,
            question=question,
            answer=answer,
            rationale=rationale,
            verdict_label=verdict_label,
            judge_feedback=judge_feedback,
            repair_priorities=repair_priorities,
        )

    def expand_input(
        self,
        *,
        question: str,
        answer: str,
        boundary: str,
        dimension: str,
        judge_feedback: str,
    ) -> str:
        return self._format(
            "expand_input",
            question=question,
            answer=answer,
            boundary=boundary,
            dimension=dimension,
            judge_feedback=judge_feedback,
        )

    def refine_input(
        self,
        *,
        question: str,
        answer: str,
        apex: str,
        apex_guidance: str,
        left_label: str,
        left_boundary: str,
        right_label: str,
        right_boundary: str,
        judge_feedback: str,
    ) -> str:
        return self._format(
            "refine_input",
            question=question,
            answer=answer,
            apex=apex,
            apex_guidance=apex_guidance,
            left_label=left_label,
            left_boundary=left_boundary,
            right_label=right_label,
            right_boundary=right_boundary,
            judge_feedback=judge_feedback,
        )

    def contradiction_check(self, *, first: str, second: str) -> str:
        return self._format("contradiction_check", first=first, second=second)

    def drift_check(self, *, question: str, answer: str) -> str:
        return self._format("drift_check", question=question, answer=answer)

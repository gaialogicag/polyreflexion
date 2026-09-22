"""The three polycontextural judges.

``PolyJudge`` evaluates one completed engine run from three logical contexts
in parallel.  Each judge replies with the proven JSON contract
(``{"value": ..., "rationale": ...}``); any parse failure or off-schema value
falls back to the conservative default for its context (subjective -> F,
objective -> F, dialectical -> R) with ``method="fallback"`` so the controller
can detect low-confidence cycles.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from polyreflexion.meta.datatypes import (
    DIMENSION_ORDER,
    ContextJudgment,
    Dimension,
    Observation,
    PolyEvaluation,
)
from polyreflexion.meta.prompts import MetaPromptRegistry
from polyreflexion.models.openai_client import parse_json_response

# Accepted labels per pole and context (upper-cased before comparison).
_POSITIVE_VALUES: dict[Dimension, set[str]] = {
    Dimension.SUBJECTIVE: {"A"},
    Dimension.OBJECTIVE: {"W"},
    Dimension.DIALECTICAL: {"W"},
}
_NEGATIVE_VALUES: dict[Dimension, set[str]] = {
    Dimension.SUBJECTIVE: {"F"},
    Dimension.OBJECTIVE: {"F"},
    Dimension.DIALECTICAL: {"R"},
}


class PolyJudge:
    """Run the subjective, objective, and dialectical judges concurrently."""

    def __init__(
        self,
        clients,  # LLMClient | dict[Dimension, LLMClient]
        prompts: MetaPromptRegistry,
        *,
        parallel: bool = True,
    ) -> None:
        # A single client is shared across all three contexts; a dict allows a
        # different backend per dimension (per the spec's outlook).
        if isinstance(clients, dict):
            self._clients = dict(clients)
        else:
            self._clients = {dim: clients for dim in DIMENSION_ORDER}
        self._prompts = prompts
        self._parallel = parallel

    def evaluate(self, observation: Observation) -> PolyEvaluation:
        """Judge one observation in all three contexts (parallel LLM calls)."""
        if self._parallel:
            with ThreadPoolExecutor(max_workers=3, thread_name_prefix="judge") as pool:
                futures = {
                    dim: pool.submit(self._judge_one, dim, observation) for dim in DIMENSION_ORDER
                }
                judgments = {dim: future.result() for dim, future in futures.items()}
        else:
            judgments = {dim: self._judge_one(dim, observation) for dim in DIMENSION_ORDER}

        return PolyEvaluation(
            subjective=judgments[Dimension.SUBJECTIVE],
            objective=judgments[Dimension.OBJECTIVE],
            dialectical=judgments[Dimension.DIALECTICAL],
        )

    def _judge_one(self, dimension: Dimension, observation: Observation) -> ContextJudgment:
        prompt = self._prompts.judge(
            dimension,
            question=observation.input_text,
            answer=observation.summary,
            corners=observation.corners_block(),
            previous_answer=observation.previous_summary or "(none)",
        )
        try:
            raw = self._clients[dimension].complete(prompt)
            data = parse_json_response(raw)
            value = str(data.get("value", "")).strip().upper()
            rationale = str(data.get("rationale", "")).strip()
            if value in _POSITIVE_VALUES[dimension]:
                return ContextJudgment.make(dimension, True, rationale)
            if value in _NEGATIVE_VALUES[dimension]:
                return ContextJudgment.make(dimension, False, rationale)
            raise ValueError(f"off-schema value {value!r}")
        except Exception as exc:
            # Conservative default: never let a broken judge look positive.
            return ContextJudgment.make(
                dimension,
                False,
                f"fallback after judge error: {exc}",
                method="fallback",
            )


class StubMetaClient:
    """Deterministic scripted stub for all meta-layer LLM calls (tests / --stub).

    ``triples`` scripts the judge verdicts: entry k is used for the k-th
    evaluation of each dimension (the last entry repeats once exhausted).
    Drift and contradiction checks are scriptable the same way.
    """

    def __init__(
        self,
        triples: list[tuple[str, str, str]] | None = None,
        *,
        drift_answers: list[str] | None = None,
        contradiction: str = "no",
    ) -> None:
        self._triples = [tuple(t) for t in (triples or [("A", "W", "R")])]
        self._counts: dict[Dimension, int] = {dim: 0 for dim in DIMENSION_ORDER}
        self._drift_answers = list(drift_answers or [])
        self._contradiction = contradiction

    def complete(self, prompt: str) -> str:
        # Marker strings match the templates in prompts_meta.json.
        if "mutually contradictory" in prompt:
            return self._contradiction
        if "still address the original question" in prompt:
            return self._drift_answers.pop(0) if self._drift_answers else "yes"
        if "separates this answer from its opposite" in prompt:
            return "[Boundary: what still separates this answer from its opposite]"
        for position, dimension, marker in (
            (0, Dimension.SUBJECTIVE, "SUBJECTIVE context"),
            (1, Dimension.OBJECTIVE, "OBJECTIVE context"),
            (2, Dimension.DIALECTICAL, "DIALECTICAL context"),
        ):
            if marker in prompt:
                index = min(self._counts[dimension], len(self._triples) - 1)
                self._counts[dimension] += 1
                value = self._triples[index][position]
                return json.dumps(
                    {"value": value, "rationale": f"stub {dimension.value} verdict {value}"}
                )
        return "[stub meta response]"

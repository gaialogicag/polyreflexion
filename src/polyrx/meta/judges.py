"""The three polycontextural judges.

``PolyJudge`` evaluates one completed engine run from three logical contexts
in parallel. A text-completion judge replies with the proven JSON contract
(``{"value": ..., "rationale": ...}``); any parse failure or off-schema value
falls back to the conservative default for its context (subjective -> F,
objective -> F, dialectical -> R) with ``method="fallback"`` so the controller
can detect low-confidence cycles. An ``AnyJevClient`` judge instead answers a
typed yes/no question directly (see ``_ANYJEV_QUESTION``), with no text to
parse and no off-schema case -- it can still fail (a missing model, an
unreachable server), which the same fallback path catches.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from polyrx.meta.datatypes import (
    DIMENSION_ORDER,
    ContextJudgment,
    Dimension,
    Observation,
    PolyEvaluation,
)
from polyrx.meta.prompts import MetaPromptRegistry
from polyrx.models.anyjev_client import AnyJevClient
from polyrx.models.openai_client import parse_json_response
from polyrx.usage import submit_in_context

#: The yes/no question each dimension's judge is actually answering, independent
#: of the dataset-specific framing in prompts_meta.json. Generic on purpose: an
#: AnyJev-backed judge asks this directly instead of parsing it back out of a
#: dataset's own prompt wording.
_ANYJEV_QUESTION: dict[Dimension, str] = {
    Dimension.SUBJECTIVE: (
        "Is this answer authentic and internally coherent, given the reasoning "
        "artifacts shown alongside it?"
    ),
    Dimension.OBJECTIVE: (
        "Is this answer objectively faithful to the original problem, with no "
        "invented or dropped facts?"
    ),
    Dimension.DIALECTICAL: (
        "Does this answer add new substance beyond the previous cycle's answer, "
        "rather than only paraphrasing it?"
    ),
}

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
                    dim: submit_in_context(pool, self._judge_one, dim, observation)
                    for dim in DIMENSION_ORDER
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
        client = self._clients[dimension]
        try:
            if isinstance(client, AnyJevClient):
                # A typed yes/no verdict, not JSON parsed out of free text: the
                # prompt itself (minus the JSON-reply instruction a text model
                # needs but AnyJev does not) is the state; the dimension's
                # generic question is asked about it directly.
                state = prompt.rsplit("Reply with JSON only:", 1)[0].strip()
                positive, p_true = client.decide_verdict(state, _ANYJEV_QUESTION[dimension])
                return ContextJudgment.make(
                    dimension,
                    positive,
                    f"AnyJev {client.config.level}: p={p_true:.3f}",
                    method=f"anyjev_{client.config.level.lower()}",
                )
            raw = client.complete(prompt)
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

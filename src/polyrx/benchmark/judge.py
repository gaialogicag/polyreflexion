"""Scoring an answer against its gold label.

One judge for every dataset. The dataset-specific part — turning a free-form
response into one of the allowed answers — belongs to the adapter; this module
only decides whether the result is right, and how confidently it knows.

Three methods, in order of preference:

``matched``
    The adapter recognised the response as one of the allowed answers, so a
    comparison settles it with no model call. This is the path almost every
    answer takes.
``llm_judge``
    The response did not land on an allowed answer and a judge template is
    configured, so a model decides. Costs a call; catches paraphrases.
``fallback_match``
    No judge template, or the judge failed. A string comparison, recorded as
    such so a reader can discount it.
"""

from __future__ import annotations

import re

from polyrx.datasets.base import DatasetAdapter, Item
from polyrx.engine import PromptRegistry
from polyrx.models.base import LLMClient
from polyrx.models.openai_client import parse_json_response


def normalize_label(text: str) -> str:
    """Normalise a label for comparison: quotes, spacing, trailing stop, case."""
    text = str(text).strip().strip('"').strip("'").strip(".")
    return re.sub(r"\s+", " ", text).lower()


def labels_match(prediction: str, gold: str) -> bool:
    """Last-resort comparison when no judge is available."""
    pred, want = normalize_label(prediction), normalize_label(gold)
    if not pred:
        return False
    return pred == want or want in pred or pred in want


class Judge:
    """Scores answers for any dataset, using its adapter to interpret them."""

    def __init__(
        self,
        client: LLMClient,
        prompts: PromptRegistry,
        adapter: DatasetAdapter,
    ) -> None:
        self._client = client
        self._prompts = prompts
        self._adapter = adapter

    def evaluate(self, *, prediction: str, item: Item) -> dict:
        """Return ``{correct, normalized_answer, method}`` for one answer."""
        matched = self._adapter.match_label(prediction, item)

        if not self._adapter.needs_judge(matched, item):
            return {
                "correct": normalize_label(matched) == normalize_label(item.gold_label),
                "normalized_answer": matched,
                "method": "matched",
            }

        if self._prompts.has_judge_template():
            try:
                raw = self._client.complete(
                    self._prompts.judge(
                        gold=item.gold_label,
                        prediction=prediction,
                        label_space=item.label_space,
                    )
                )
                data = parse_json_response(raw)
                return {
                    "correct": bool(data.get("correct", False)),
                    "normalized_answer": str(data.get("normalized_answer", matched)),
                    "method": "llm_judge",
                }
            except Exception:
                pass

        return {
            "correct": labels_match(prediction, item.gold_label),
            "normalized_answer": matched.strip(),
            "method": "fallback_match",
        }

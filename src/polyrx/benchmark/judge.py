"""LLM-as-judge scoring for OpenToM answers."""

from __future__ import annotations

import re

from polyrx.engine import PromptRegistry
from polyrx.models.base import LLMClient
from polyrx.models.openai_client import parse_json_response


def normalize_label(text: str) -> str:
    """Normalize a label string for comparison."""
    text = text.strip().strip('"').strip("'").strip(".")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def labels_match(prediction: str, gold: str, label_space: str) -> bool:
    """Programmatic fallback when judge JSON parsing fails."""
    pred = normalize_label(prediction)
    gold_norm = normalize_label(gold)
    if pred == gold_norm:
        return True
    allowed = [normalize_label(part) for part in label_space.split(",")]
    if pred in allowed and pred == gold_norm:
        return True
    return pred == gold_norm or gold_norm in pred or pred in gold_norm


class OpenToMJudge:
    """Score model answers against gold labels using an LLM judge."""

    def __init__(self, client: LLMClient, prompts: PromptRegistry) -> None:
        self._client = client
        self._prompts = prompts

    def evaluate(
        self,
        *,
        gold: str,
        prediction: str,
        label_space: str,
    ) -> dict:
        """Return verdict dict with correct, normalized_answer, and method."""
        prompt = self._prompts.opentom_judge(
            gold=gold,
            prediction=prediction,
            label_space=label_space,
        )
        try:
            raw = self._client.complete(prompt)
            data = parse_json_response(raw)
            correct = bool(data.get("correct", False))
            normalized = str(data.get("normalized_answer", prediction))
            return {
                "correct": correct,
                "normalized_answer": normalized,
                "method": "llm_judge",
            }
        except Exception:
            correct = labels_match(prediction, gold, label_space)
            return {
                "correct": correct,
                "normalized_answer": prediction.strip(),
                "method": "fallback_match",
            }

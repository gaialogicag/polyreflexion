"""Letter-first scoring for GPQA Diamond answers."""

from __future__ import annotations

import re

from polyreflexion.benchmark.judge import labels_match, normalize_label
from polyreflexion.engine import PromptRegistry
from polyreflexion.models.base import LLMClient
from polyreflexion.models.openai_client import parse_json_response

_LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_choice_letter(prediction: str, label_space: str = "A, B, C, D") -> str:
    """Normalize a free-form prediction to a single A–D letter when possible."""
    allowed = {normalize_label(p) for p in label_space.split(",") if p.strip()}
    cleaned = prediction.strip()
    # Prefer \boxed{X} / final short lines.
    boxed = re.search(r"\\boxed\{\s*([A-Da-d])\s*\}", cleaned)
    if boxed:
        return boxed.group(1).upper()
    lines = [ln.strip().strip('"').strip("'") for ln in cleaned.splitlines() if ln.strip()]
    for candidate in reversed(lines) if lines else [cleaned]:
        key = normalize_label(candidate)
        if key in allowed:
            return key.upper()
        match = _LETTER_RE.search(candidate)
        if match and normalize_label(match.group(1)) in allowed:
            return match.group(1).upper()
    match = _LETTER_RE.search(cleaned)
    if match and normalize_label(match.group(1)) in allowed:
        return match.group(1).upper()
    return cleaned.splitlines()[-1].strip() if lines else cleaned


class GPQAJudge:
    """Score GPQA answers: exact letter match first, LLM judge as fallback."""

    def __init__(self, client: LLMClient, prompts: PromptRegistry | None = None) -> None:
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
        gold_letter = normalize_label(gold).upper()
        pred_letter = extract_choice_letter(prediction, label_space)
        pred_norm = normalize_label(pred_letter).upper()
        if pred_norm in {"A", "B", "C", "D"} and pred_norm == gold_letter:
            return {
                "correct": True,
                "normalized_answer": pred_norm,
                "method": "exact_letter",
            }
        if pred_norm in {"A", "B", "C", "D"} and pred_norm != gold_letter:
            return {
                "correct": False,
                "normalized_answer": pred_norm,
                "method": "exact_letter",
            }

        # Ambiguous free text: optional LLM judge, then string fallback.
        if self._prompts is not None and getattr(self._prompts, "_gpqa_judge", ""):
            try:
                prompt = self._prompts.gpqa_judge(
                    gold=gold,
                    prediction=prediction,
                    label_space=label_space,
                )
                raw = self._client.complete(prompt)
                data = parse_json_response(raw)
                correct = bool(data.get("correct", False))
                normalized = str(data.get("normalized_answer", pred_letter))
                return {
                    "correct": correct,
                    "normalized_answer": normalized,
                    "method": "llm_judge",
                }
            except Exception:
                pass

        correct = labels_match(prediction, gold, label_space)
        return {
            "correct": correct,
            "normalized_answer": pred_letter.strip(),
            "method": "fallback_match",
        }

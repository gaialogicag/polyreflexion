"""GPQA-specific qualitative analyses (domain errors, scientific markers)."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from polyreflexion.benchmark.qualitative_analysis import (
    FlipRecord,
    _judgment_correct,
    _prediction,
    collect_flips,
)

# Heuristic markers for explicit scientific reasoning in integrated summaries.
UNCERTAINTY_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\buncertain(?:ty)?\b",
        r"\bassump(?:tion|e|es|ing)\b",
        r"\bif and only if\b",
        r"\bapprox(?:imate|imately)?\b",
        r"\bneglect(?:ing|s)?\b",
        r"\blimiting case\b",
        r"\bunder (?:these|those) conditions\b",
    )
)

MECHANISM_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmechanism\b",
        r"\bbecause\b",
        r"\btherefore\b",
        r"\bimply(?:ing|ies)?\b",
        r"\bgovern(?:ed|s|ing)?\b",
        r"\bprinciple\b",
        r"\blaw of\b",
        r"\brate[- ]limiting\b",
        r"\bequilibrium\b",
        r"\bconservation of\b",
    )
)

DISTRACTOR_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bdistractor\b",
        r"\beliminat(?:e|es|ing|ion)\b",
        r"\brule(?:s|d)? out\b",
        r"\bincorrect (?:because|since|option)\b",
        r"\btempting\b",
        r"\bcommon (?:mistake|error|trap)\b",
        r"\bnot (?:A|B|C|D)\b",
        r"\bwrong (?:because|since)\b",
    )
)

CALCULATION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bcalculat(?:e|es|ing|ion)\b",
        r"\bequat(?:ion|ions)\b",
        r"\bderiv(?:e|es|ing|ation)\b",
        r"\bunit(?:s)?\b",
        r"\border of magnitude\b",
        r"\bproportional(?:ity)?\b",
        r"[=≈∼~]",
        r"\b\d+\s*(?:eV|nm|kJ|kcal|Hz|MHz|GHz|mol|M)\b",
    )
)

MARKER_SETS: dict[str, tuple[re.Pattern[str], ...]] = {
    "uncertainty": UNCERTAINTY_PATTERNS,
    "mechanism": MECHANISM_PATTERNS,
    "distractor_elimination": DISTRACTOR_PATTERNS,
    "calculation": CALCULATION_PATTERNS,
}


def domain_error_analysis(
    results: list[dict],
    *,
    baseline: str,
    condition: str,
) -> dict[str, dict[str, int | float]]:
    """Per-domain wrong→right / right→wrong counts vs baseline."""
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for row in results:
        domain = row.get("question_type") or "unspecified"
        by_domain[domain].append(row)

    out: dict[str, dict[str, int | float]] = {}
    for domain, rows in sorted(by_domain.items()):
        flip = collect_flips(rows, baseline=baseline, condition=condition)
        base_wrong = cond_wrong = both_wrong = 0
        for row in rows:
            base_ok = _judgment_correct(row, baseline)
            cond_ok = _judgment_correct(row, condition)
            if base_ok is None or cond_ok is None:
                continue
            if not base_ok:
                base_wrong += 1
            if not cond_ok:
                cond_wrong += 1
            if not base_ok and not cond_ok:
                both_wrong += 1
        total = len(rows)
        out[domain] = {
            "total": total,
            "baseline_wrong": base_wrong,
            "condition_wrong": cond_wrong,
            "fixed": len(flip.gains),
            "regressed": len(flip.losses),
            "both_wrong": both_wrong,
            "baseline_error_rate": base_wrong / total if total else 0.0,
            "condition_error_rate": cond_wrong / total if total else 0.0,
        }
    return out


def flip_examples_by_domain(
    results: list[dict],
    *,
    baseline: str,
    condition: str,
) -> dict[str, list[FlipRecord]]:
    """Group gain flips by domain for example tables."""
    flip = collect_flips(results, baseline=baseline, condition=condition)
    by_domain: dict[str, list[FlipRecord]] = defaultdict(list)
    for rec in flip.gains:
        by_domain[rec.question_type or "unspecified"].append(rec)
    return dict(by_domain)


def count_prediction_letters(results: list[dict], condition: str) -> Counter[str]:
    """Distribution of predicted letters for one condition."""
    counts: Counter[str] = Counter()
    for row in results:
        pred = _prediction(row, condition).strip().upper()
        if pred in {"A", "B", "C", "D"}:
            counts[pred] += 1
        elif pred:
            counts["other"] += 1
    return counts

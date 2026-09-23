"""Qualitative analyses for a benchmark run (flips, summaries, markers)."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Summary cache keys aligned with Condition.summary_key in polyrx.conditions.
REFLEXION_SUMMARY_KEY = "answerer"
META_SUMMARY_KEYS = ("answerer_meta_c1", "answerer_meta_c2", "answerer_meta_c3", "answerer_meta_c4")

# Question types where wrong answers usually reflect faulty ToM inference.
INFERENCE_QUESTION_TYPES = frozenset({"location-fo", "location-so", "multihop-fo", "multihop-so"})

# Heuristic markers for explicit false-belief / misread awareness in summaries.
FALSE_BELIEF_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmistaken(?:ly)?\b",
        r"\bfalse belief\b",
        r"\bincorrect(?:ly)? believes?\b",
        r"\bwrong(?:ly)? believes?\b",
        r"\bmisunderstanding\b",
        r"\bdiffers from reality\b",
        r"\bbelief that differed\b",
        r"\bnot shown as\b",
        r"\bmisread(?:ing)?\b",
        r"\bunaware that\b",
        r"\bdoes not know\b",
        r"\bremains unaware\b",
        r"\bstill thinks\b",
        r"\bbelieves .{0,40} (?:still|where they (?:last )?saw)\b",
    )
)

# Heuristic markers for contradiction / tension detection in summaries.
CONTRADICTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bcontradict",
        r"\bconflict(?:ed|ing)?\b",
        r"\binconsistent\b",
        r"\buneasy\b",
        r"\bconflicted\b",
        r"\btension\b",
        r"\bdisbelief\b",
        r"\bopposite (?:beliefs?|preferences?|feelings?)\b",
        r"\bbelief differs\b",
        r"\bdiffer(?:s|ed) from (?:reality|what)\b",
        r"\beven though\b",
    )
)


@dataclass(frozen=True)
class FlipRecord:
    """One question where accuracy changed vs a baseline condition."""

    item_id: str
    question: str
    group: str
    gold_label: str
    baseline_pred: str
    condition_pred: str


@dataclass
class FlipSummary:
    """Aggregated flip counts and example rows for one condition."""

    condition: str
    baseline: str
    gains: list[FlipRecord] = field(default_factory=list)
    losses: list[FlipRecord] = field(default_factory=list)

    @property
    def net(self) -> int:
        return len(self.gains) - len(self.losses)


@dataclass
class SummaryLengthStats:
    """Word/character length stats for one summary cache key."""

    key: str
    story_count: int
    mean_words: float
    mean_chars: float
    total_words: int


@dataclass
class MarkerStoryHit:
    """One story where heuristic markers fired in a summary."""

    item_id: str
    summary_key: str
    markers: tuple[str, ...]
    excerpt: str


def _judgment_correct(row: dict, condition: str) -> bool | None:
    judgment = row.get("judgments", {}).get(condition)
    if not judgment:
        return None
    return bool(judgment.get("correct", False))


def _prediction(row: dict, condition: str) -> str:
    return str(row.get("predictions", {}).get(condition, ""))


def collect_flips(
    results: list[dict],
    *,
    baseline: str,
    condition: str,
) -> FlipSummary:
    """Collect wrong→right (gains) and right→wrong (losses) vs baseline."""
    gains: list[FlipRecord] = []
    losses: list[FlipRecord] = []
    for row in results:
        base_ok = _judgment_correct(row, baseline)
        cond_ok = _judgment_correct(row, condition)
        if base_ok is None or cond_ok is None or base_ok == cond_ok:
            continue
        record = FlipRecord(
            item_id=row["item_id"],
            question=row["question"],
            group=row.get("group", ""),
            gold_label=row.get("gold_label", ""),
            baseline_pred=_prediction(row, baseline),
            condition_pred=_prediction(row, condition),
        )
        if cond_ok and not base_ok:
            gains.append(record)
        else:
            losses.append(record)
    return FlipSummary(condition=condition, baseline=baseline, gains=gains, losses=losses)


def inference_error_analysis(
    results: list[dict],
    *,
    baseline: str,
    condition: str,
) -> dict[str, int | float]:
    """Compare erroneous ToM inference (wrong on location/multihop types)."""
    scoped = [r for r in results if r.get("group") in INFERENCE_QUESTION_TYPES]
    base_wrong = cond_wrong = 0
    fixed = new_errors = persistent_same = changed_error = 0
    for row in scoped:
        base_ok = _judgment_correct(row, baseline)
        cond_ok = _judgment_correct(row, condition)
        if base_ok is None or cond_ok is None:
            continue
        if not base_ok:
            base_wrong += 1
        if not cond_ok:
            cond_wrong += 1
        if not base_ok and cond_ok:
            fixed += 1
        elif base_ok and not cond_ok:
            new_errors += 1
        elif not base_ok and not cond_ok:
            bp = _prediction(row, baseline)
            cp = _prediction(row, condition)
            if bp == cp:
                persistent_same += 1
            else:
                changed_error += 1
    total = len(scoped)
    return {
        "total_inference_questions": total,
        "baseline_wrong": base_wrong,
        "condition_wrong": cond_wrong,
        "fixed_from_baseline": fixed,
        "new_errors_vs_baseline": new_errors,
        "both_wrong_same_answer": persistent_same,
        "both_wrong_different_answer": changed_error,
        "baseline_error_rate": base_wrong / total if total else 0.0,
        "condition_error_rate": cond_wrong / total if total else 0.0,
    }


def summary_length_stats(
    summaries: dict[str, dict[str, str]],
    summary_key: str,
) -> SummaryLengthStats | None:
    """Mean summary length for one cache key across stories."""
    texts: list[str] = []
    for story_summaries in summaries.values():
        text = story_summaries.get(summary_key, "").strip()
        if text:
            texts.append(text)
    if not texts:
        return None
    words = [len(t.split()) for t in texts]
    chars = [len(t) for t in texts]
    return SummaryLengthStats(
        key=summary_key,
        story_count=len(texts),
        mean_words=sum(words) / len(words),
        mean_chars=sum(chars) / len(chars),
        total_words=sum(words),
    )


def _find_markers(text: str, patterns: tuple[re.Pattern[str], ...]) -> tuple[str, ...]:
    hits: list[str] = []
    for pattern in patterns:
        if pattern.search(text):
            hits.append(pattern.pattern)
    return tuple(hits)


def stories_with_markers(
    summaries: dict[str, dict[str, str]],
    summary_key: str,
    patterns: tuple[re.Pattern[str], ...],
    *,
    excerpt_chars: int = 160,
) -> list[MarkerStoryHit]:
    """Return stories whose summary matches at least one heuristic marker."""
    hits: list[MarkerStoryHit] = []
    for item_id, story_summaries in sorted(summaries.items()):
        text = story_summaries.get(summary_key, "").strip()
        if not text:
            continue
        markers = _find_markers(text, patterns)
        if not markers:
            continue
        hits.append(
            MarkerStoryHit(
                item_id=item_id,
                summary_key=summary_key,
                markers=markers,
                excerpt=text[:excerpt_chars].replace("\n", " ")
                + ("…" if len(text) > excerpt_chars else ""),
            )
        )
    return hits


def meta_adds_markers_vs_reflexion(
    summaries: dict[str, dict[str, str]],
    meta_key: str,
    *,
    reflexion_key: str = REFLEXION_SUMMARY_KEY,
    patterns: tuple[re.Pattern[str], ...] = FALSE_BELIEF_PATTERNS,
) -> list[MarkerStoryHit]:
    """Stories where meta summary has markers that plain reflexion summary lacks."""
    added: list[MarkerStoryHit] = []
    for item_id, story_summaries in sorted(summaries.items()):
        reflex = story_summaries.get(reflexion_key, "")
        meta = story_summaries.get(meta_key, "")
        if not meta.strip():
            continue
        meta_markers = _find_markers(meta, patterns)
        if not meta_markers:
            continue
        reflex_markers = _find_markers(reflex, patterns)
        if set(meta_markers) - set(reflex_markers) or (meta_markers and not reflex_markers):
            added.append(
                MarkerStoryHit(
                    item_id=item_id,
                    summary_key=meta_key,
                    markers=meta_markers,
                    excerpt=meta[:160].replace("\n", " ") + ("…" if len(meta) > 160 else ""),
                )
            )
    return added


def count_by_type(records: list[FlipRecord]) -> Counter[str]:
    return Counter(r.group for r in records)


def _escape_cell(text: str, max_len: int = 72) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text

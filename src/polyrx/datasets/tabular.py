"""The adapter most datasets need, driven entirely from configuration.

Reads CSV, TSV, JSON and JSONL. A source is described by ``dataset.fields``:
which column is the question, which is the answer, optionally which carries the
context, the grouping dimension, and the multiple-choice options. No Python is
needed to add a dataset in any of these shapes.

Two choice layouts are supported, because public datasets use both:

* options in their own columns (``choices: [option_a, option_b, ...]``), with
  the answer column naming the correct letter;
* a correct answer column plus incorrect answer columns
  (``correct_choice`` / ``incorrect_choices``), which are shuffled into a
  stable per-item order.
"""

from __future__ import annotations

import ast
import contextlib
import csv
import json
import random
import re
from pathlib import Path
from typing import Any

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter

#: Letters assigned to multiple-choice options, in order.
CHOICE_LETTERS = tuple("ABCDEFGH")

_WHITESPACE = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Normalise an answer for comparison: case, spacing and trailing marks."""
    return _WHITESPACE.sub(" ", str(text)).strip().strip(".").casefold()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    """Read a source file into rows, whatever tabular format it uses."""
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            # A single object wrapping the rows under some key.
            for value in data.values():
                if isinstance(value, list):
                    return value
            return [data]
        return data
    delimiter = "\t" if suffix in {".tsv", ".tab"} else ","
    return list(csv.DictReader(text.splitlines(), delimiter=delimiter))


@register_adapter
class TabularAdapter(DatasetAdapter):
    """Config-driven adapter for row-shaped sources."""

    name = "tabular"

    # -- loading ------------------------------------------------------------

    def load(self, path: Path) -> list[Item]:
        rows = _read_rows(path)
        items: list[Item] = []
        for index, row in enumerate(rows):
            item = self._item(row, index)
            if item is not None:
                items.append(item)
        if not items:
            raise ValueError(
                f"No items parsed from {path}. Check dataset.fields against the "
                f"source's columns: {sorted(rows[0]) if rows else 'file is empty'}"
            )
        return items

    def _item(self, row: dict[str, Any], index: int) -> Item | None:
        fields = self.config.fields
        question = str(row.get(fields.question, "") or "").strip()
        answer = str(row.get(fields.answer, "") or "").strip()
        if not question or not answer:
            return None

        context = str(row.get(fields.context, "") or "").strip() if fields.context else ""
        choices = self._choices(row, question, answer)

        if choices:
            gold = self._gold_letter(row, answer, choices)
            label_space = ", ".join(CHOICE_LETTERS[: len(choices)])
            prompt_text = self._render_choices(question, choices)
        else:
            gold = answer
            label_space = self._label_space(row)
            prompt_text = question

        group = str(row.get(fields.group, "") or "").strip() if fields.group else ""
        return Item(
            item_id=Item.content_id(context, question, answer, id_chars=self.config.id_hash_chars),
            context=context or question,
            prompt_text=prompt_text,
            question=question,
            gold_label=gold,
            label_space=label_space,
            group=group or "unspecified",
            metadata={"choices": choices} if choices else {},
        )

    # -- choices ------------------------------------------------------------

    def _choices(self, row: dict[str, Any], question: str, answer: str) -> list[str]:
        """Option texts in presentation order, or empty for a free-text answer."""
        fields = self.config.fields
        if fields.choices:
            return [str(row.get(c, "") or "").strip() for c in fields.choices if row.get(c)]
        if fields.correct_choice and fields.incorrect_choices:
            correct = str(row.get(fields.correct_choice, "") or "").strip()
            wrong = [
                str(row.get(c, "") or "").strip()
                for c in fields.incorrect_choices
                if str(row.get(c, "") or "").strip()
            ]
            if not correct or not wrong:
                return []
            options = [correct, *wrong]
            # Shuffle per item, deterministically: the correct answer must not
            # always be first, and the order must not change between runs.
            digest = Item.content_id(question, correct, id_chars=8)
            random.Random(int(digest, 16)).shuffle(options)
            return options
        return []

    def _gold_letter(self, row: dict[str, Any], answer: str, choices: list[str]) -> str:
        """The letter of the correct option, however the source expressed it."""
        fields = self.config.fields
        if fields.correct_choice:
            correct = str(row.get(fields.correct_choice, "") or "").strip()
            for letter, text in zip(CHOICE_LETTERS, choices, strict=False):
                if _norm(text) == _norm(correct):
                    return letter
        stripped = answer.strip().strip("()").upper()
        if stripped in CHOICE_LETTERS[: len(choices)]:
            return stripped
        for letter, text in zip(CHOICE_LETTERS, choices, strict=False):
            if _norm(text) == _norm(answer):
                return letter
        raise ValueError(f"Cannot identify the correct option for answer {answer!r}")

    def _render_choices(self, question: str, choices: list[str]) -> str:
        lines = [question.strip(), "", "Choices:"]
        lines += [
            f"({letter}) {text}" for letter, text in zip(CHOICE_LETTERS, choices, strict=False)
        ]
        return "\n".join(lines)

    def _label_space(self, row: dict[str, Any]) -> str:
        fields = self.config.fields
        if fields.label_space and row.get(fields.label_space):
            raw = row[fields.label_space]
            if isinstance(raw, str) and raw.strip().startswith("["):
                # A list written as a string, which CSV sources often do.
                with contextlib.suppress(SyntaxError, ValueError):
                    raw = ast.literal_eval(raw)
            if isinstance(raw, list | tuple):
                return ", ".join(str(x).strip() for x in raw)
            return str(raw)
        return self.config.label_space

    # -- scoring ------------------------------------------------------------

    @staticmethod
    def _match_letter(cleaned: str, lines: list[str], allowed: list[str]) -> str:
        """Find a choice letter, but only where one is actually being given.

        A bare word-boundary search is not safe here: the English article in
        "a cat sat" is not an answer of "A". Only letters in answer-shaped
        positions count — a line that is just the letter, a boxed answer, a
        parenthesised or bolded letter, or one introduced by "answer".
        Anything else comes back unmatched so the judge sees it.
        """
        letters = "".join(re.escape(a) for a in allowed)
        patterns = (
            rf"^\W*([{letters}])\W*$",  # the whole line is the letter
            rf"\\boxed\{{\s*([{letters}])\s*\}}",  # \boxed{{X}}
            rf"\*\*\s*\(?([{letters}])\)?[.:)]?\s*\*\*",  # **X**
            rf"\(\s*([{letters}])\s*\)",  # (X)
            rf"answer\b\W*(?:is\b\W*)?([{letters}])\b",  # Answer: X / answer is X
            # An explicit choice verb. Without one, a letter mentioned in
            # passing ("A is wrong") would be read as the answer.
            rf"\b(?:choose|select|pick|option)\b\W*([{letters}])\b",
            rf"^([{letters}])[.):]\s",  # X. at the start of a line
        )
        for pattern in patterns:
            for candidate in [*reversed(lines), cleaned]:
                match = re.search(pattern, candidate, re.IGNORECASE | re.MULTILINE)
                if match:
                    return match.group(1).upper()
        return cleaned

    def match_label(self, prediction: str, item: Item) -> str:
        """Map a response onto an allowed label.

        Prefers the last line that is an allowed label outright, then the last
        allowed label mentioned anywhere. Longer labels win over shorter ones
        so that ``"more full"`` is not swallowed by ``"full"``.
        """
        allowed = [p.strip() for p in item.label_space.split(",") if p.strip()]
        if not allowed:
            return prediction.strip()

        by_norm = {_norm(a): a for a in allowed}
        longest_first = sorted(allowed, key=len, reverse=True)
        cleaned = re.sub(r"^\s*Answer:\s*", "", prediction.strip(), flags=re.IGNORECASE)

        lines = [ln.strip().strip('"').strip("'") for ln in cleaned.splitlines() if ln.strip()]
        for candidate in [*reversed(lines), cleaned]:
            if _norm(candidate) in by_norm:
                return by_norm[_norm(candidate)]

        if all(len(a) == 1 for a in allowed):
            return self._match_letter(cleaned, lines, allowed)

        lowered = cleaned.casefold()
        best, best_pos = None, -1
        for label in longest_first:
            pos = lowered.rfind(_norm(label))
            if pos > best_pos:
                best, best_pos = label, pos
        return best if best is not None else cleaned

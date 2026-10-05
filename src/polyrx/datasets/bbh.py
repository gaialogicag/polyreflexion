"""BIG-Bench Hard (BBH): 23 bundled mini-benchmarks sharing one of three
answer shapes, not 23 distinct formats. This adapter branches on the shape
of each row's ``target``, not on which of the 23 tasks is loaded, so it
works unmodified for any of them:

* multiple choice, with ``"Options:\\n(A) ...\\n(B) ...\\n"`` embedded
  directly inside ``input`` and ``target`` a parenthesised letter -- e.g.
  ``date_understanding``, ``logical_deduction_*``,
  ``tracking_shuffled_objects_*``, ``temporal_sequences``;
* a fixed two-way label (True/False, Yes/No) -- e.g.
  ``boolean_expressions``, ``causal_judgement``, ``navigate``,
  ``sports_understanding``, ``web_of_lies``;
* free-form text with no fixed label set at all -- e.g. ``word_sorting``,
  ``multistep_arithmetic_two``, ``dyck_languages``, ``object_counting``.

Each ``conf/dataset/bbh_<task>.yaml`` names one HF config (one parquet
file, ``<task>/test-00000-of-00001.parquet``); this adapter does not need a
new file or a code change to support another task, only a new dataset
config pointing at a different file.

The free-form branch carries the same leak risk MathArena's integer answer
does: the gold value cannot be stated in the answerer's own prompt without
handing it the answer. ``conf/prompts/reflexion/bbh_freeform.yaml``'s `qa`
template is written to never render ``{label_space}`` for exactly that
reason; the MCQ and two-way branches use the ordinary ``default`` prompt
set, where showing the label space is safe because it does not say which
one is correct.
"""

from __future__ import annotations

import re
from pathlib import Path

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter
from polyrx.datasets.tabular import TabularAdapter

_OPTIONS_RE = re.compile(r"\n?Options:\s*\n((?:\([A-Z]\)[^\n]*\n?)+)")
_OPTION_LINE_RE = re.compile(r"\(([A-Z])\)\s*(.+)")
_PAREN_LETTER_RE = re.compile(r"^\(([A-Z])\)$")
_PUNCT_RE = re.compile(r"[,;]+")
_WHITESPACE_RE = re.compile(r"\s+")

#: Target casefold -> canonical label space, preserving the source's own
#: capitalization (confirmed from the actual data: "True"/"False",
#: "Yes"/"No", never lowercase).
_TWO_WAY_LABEL_SPACES = {
    frozenset({"true", "false"}): "True, False",
    frozenset({"yes", "no"}): "Yes, No",
}


@register_adapter
class BBHAdapter(DatasetAdapter):
    name = "bbh"

    def load(self, path: Path) -> list[Item]:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        inputs = table.column("input").to_pylist()
        targets = table.column("target").to_pylist()
        task = self.config.name

        items: list[Item] = []
        for raw_input, target in zip(inputs, targets, strict=True):
            raw_input = (raw_input or "").strip()
            target = (target or "").strip()
            if not raw_input or not target:
                continue
            items.append(self._item(task, raw_input, target))
        if not items:
            raise ValueError(f"No items parsed from {path}")
        return items

    def _item(self, task: str, raw_input: str, target: str) -> Item:
        item_id = Item.content_id(raw_input, target, id_chars=self.config.id_hash_chars)

        letter_match = _PAREN_LETTER_RE.match(target)
        options_match = _OPTIONS_RE.search(raw_input)
        if letter_match and options_match:
            return self._mcq_item(task, item_id, raw_input, letter_match.group(1), options_match)

        two_way = self._two_way_label_space(target)
        if two_way is not None:
            return Item(
                item_id=item_id,
                context=raw_input,
                prompt_text=raw_input,
                question=raw_input,
                gold_label=target,
                label_space=two_way,
                group=task,
                metadata={"shape": "two_way"},
            )

        return Item(
            item_id=item_id,
            context=raw_input,
            prompt_text=raw_input,
            question=raw_input,
            gold_label=target,
            # A single value, internal to scoring only -- see the module
            # docstring and conf/prompts/reflexion/bbh_freeform.yaml for why
            # this must never reach the answerer's own prompt.
            label_space=target,
            group=task,
            metadata={"shape": "freeform"},
        )

    def _mcq_item(
        self, task: str, item_id: str, raw_input: str, gold_letter: str, options_match: re.Match
    ) -> Item:
        stem = raw_input[: options_match.start()].strip()
        choices: dict[str, str] = {}
        for line in options_match.group(1).strip().splitlines():
            option = _OPTION_LINE_RE.match(line.strip())
            if not option:
                raise ValueError(f"Could not parse option line {line!r} in task {task!r}")
            choices[option.group(1)] = option.group(2).strip()
        if gold_letter not in choices:
            raise ValueError(f"Gold letter {gold_letter!r} not among parsed choices {choices!r}")
        return Item(
            item_id=item_id,
            context=raw_input,
            prompt_text=raw_input,
            question=stem,
            gold_label=gold_letter,
            label_space=", ".join(sorted(choices)),
            group=task,
            metadata={"choices": choices, "shape": "mcq"},
        )

    @staticmethod
    def _two_way_label_space(target: str) -> str | None:
        lowered = target.casefold()
        for pair, label_space in _TWO_WAY_LABEL_SPACES.items():
            if lowered in pair:
                return label_space
        return None

    # -- scoring --------------------------------------------------------------

    def match_label(self, prediction: str, item: Item) -> str:
        if item.metadata.get("shape") == "freeform":
            gold_norm = self._normalize_freeform(item.gold_label)
            lines = [line.strip() for line in prediction.strip().splitlines() if line.strip()]
            for candidate in (*reversed(lines), prediction.strip()):
                if self._normalize_freeform(candidate) == gold_norm:
                    # Punctuation/case differences only (e.g. a comma-joined
                    # list where the gold is space-joined): return the gold
                    # string itself so the base class's own needs_judge sees
                    # an exact match and never calls the judge for something
                    # a plain string comparison already settles. A judge
                    # call on this exact task shape (ordered-list exact
                    # match) is demonstrably less reliable than a
                    # programmatic check -- see the smoke test that found
                    # this: a correctly-ordered, comma-punctuated answer was
                    # marked wrong, and a wrongly-ordered one was marked
                    # right, on the same two-item sample.
                    return item.gold_label
        return TabularAdapter(self.config).match_label(prediction, item)

    @staticmethod
    def _normalize_freeform(text: str) -> str:
        text = _PUNCT_RE.sub(" ", text.strip().casefold())
        return _WHITESPACE_RE.sub(" ", text).strip()

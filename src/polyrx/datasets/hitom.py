"""Hi-ToM: higher-order false-belief reasoning, 15-way multiple choice.

Needs its own adapter for two reasons the tabular one cannot cover:

* the source's ``choices`` field is a single pre-rendered string
  (``"A. foo, B. bar, ..."``), not one column per option and not a
  correct/incorrect-column pair, so it does not match either choice layout
  the tabular adapter understands;
* belief-nesting order (``question_order``, 0-4) is the entire reason this
  dataset was proposed -- H6 and the "Proposed datasets" paragraph both name
  it as a scalar, measurable variable -- so it is the grouping dimension,
  not an ignored field.

The same story is asked about at every order from 0 (the plain fact: where
is the object really) to 4 (a four-deep nested belief: what does D think C
thinks B thinks A thinks), so items are sampled by shared story
(``sample_by: context``), the same as OpenToM and ExploreToM.
"""

from __future__ import annotations

import json
from pathlib import Path

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter
from polyrx.datasets.tabular import CHOICE_LETTERS, TabularAdapter


@register_adapter
class HiToMAdapter(DatasetAdapter):
    name = "hitom"

    def load(self, path: Path) -> list[Item]:
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not rows:
            raise ValueError(f"No rows in {path}")
        return [self._item(row) for row in rows]

    def _item(self, row: dict) -> Item:
        story = (row.get("story") or "").strip()
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not story or not question or not answer:
            raise ValueError(f"Hi-ToM row is missing a story, question, or answer: {row!r}")

        options = self._parse_choices(row.get("choices") or "")
        if len(options) > len(CHOICE_LETTERS):
            raise ValueError(
                f"{len(options)} choices exceeds the {len(CHOICE_LETTERS)} letters "
                f"CHOICE_LETTERS supports"
            )
        gold = next(
            (
                letter
                for letter, text in zip(CHOICE_LETTERS, options, strict=False)
                if text == answer
            ),
            None,
        )
        if gold is None:
            raise ValueError(f"Answer {answer!r} not found among choices {options!r}")

        order = row.get("question_order")
        return Item(
            item_id=Item.content_id(story, question, answer, id_chars=self.config.id_hash_chars),
            context=story,
            prompt_text=self._render(story, question, options),
            question=question,
            gold_label=gold,
            label_space=", ".join(CHOICE_LETTERS[: len(options)]),
            group=f"order {order}" if order is not None else "unspecified",
            metadata={
                "choices": dict(zip(CHOICE_LETTERS, options, strict=False)),
                "question_order": order,
                "prompting_type": row.get("prompting_type"),
                "deception": row.get("deception"),
                "story_length": row.get("story_length"),
                "sample_id": row.get("sample_id"),
            },
        )

    @staticmethod
    def _parse_choices(raw: str) -> list[str]:
        """Split ``"A. foo, B. bar, ..."`` into option texts, in order.

        A plain split on ``", "`` is safe here: every option is a single
        underscore-joined token (``green_drawer``), never containing a comma
        of its own, across the whole source file.
        """
        options = []
        for part in raw.split(", "):
            part = part.strip()
            if len(part) > 2 and part[1] == "." and part[0].isalpha():
                options.append(part[2:].strip())
        return options

    @staticmethod
    def _render(story: str, question: str, choices: list[str]) -> str:
        lines = [story.strip(), "", question.strip(), "", "Choices:"]
        lines += [
            f"({letter}) {text}" for letter, text in zip(CHOICE_LETTERS, choices, strict=False)
        ]
        return "\n".join(lines)

    # -- scoring ------------------------------------------------------------

    def match_label(self, prediction: str, item: Item) -> str:
        return TabularAdapter(self.config).match_label(prediction, item)

"""MMLU-Pro: multiple-choice questions across 14 subjects, up to ten options.

Needs its own adapter because the source is a parquet file whose options sit
in a single list column, and the number of options varies per question (3 to
10). The tabular adapter reads one column per option and cannot do that. The
label space is built per item from its own option count, so a question with
four options never offers the indices 4-9.

The model is asked for the zero-based index of the correct option, the same
value as the source's ``answer_index`` column, and the options are shown with
those indices instead of letters.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter
from polyrx.datasets.tabular import CHOICE_LETTERS, TabularAdapter

#: A boxed integer, the form the prompt asks for.
_BOXED_INDEX_RE = re.compile(r"\\boxed\{\s*(\d+)\s*\}")


@register_adapter
class MMLUProAdapter(DatasetAdapter):
    name = "mmlu_pro"

    def load(self, path: Path) -> list[Item]:
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        columns = {"question", "options", "answer_index", "category"}
        missing = columns - set(table.column_names)
        if missing:
            raise ValueError(f"{path} is missing columns {sorted(missing)}: {table.column_names}")

        questions = table.column("question").to_pylist()
        options = table.column("options").to_pylist()
        answer_indices = table.column("answer_index").to_pylist()
        categories = table.column("category").to_pylist()

        items: list[Item] = []
        for stem, choices, answer_index, category in zip(
            questions, options, answer_indices, categories, strict=True
        ):
            stem = (stem or "").strip()
            choices = [str(c).strip() for c in choices or []]
            if (
                not stem
                or len(choices) < 2
                or answer_index is None
                or not 0 <= int(answer_index) < len(choices)
            ):
                continue
            gold = str(int(answer_index))
            items.append(
                Item(
                    # Built from the option's letter, as before the answer became an
                    # index, so summaries already cached for these items still match.
                    item_id=Item.content_id(
                        stem,
                        CHOICE_LETTERS[int(gold)],
                        *choices,
                        id_chars=self.config.id_hash_chars,
                    ),
                    # No options here: the runner appends them to every
                    # answering prompt from ``metadata["choices"]``, so putting
                    # them in the context too would show them twice.
                    context=stem,
                    prompt_text=self._render(stem, choices),
                    question=stem,
                    gold_label=gold,
                    label_space=", ".join(str(i) for i in range(len(choices))),
                    group=(category or "").strip() or "unspecified",
                    metadata={"choices": {str(i): text for i, text in enumerate(choices)}},
                )
            )
        if not items:
            raise ValueError(f"No items parsed from {path}")
        return items

    @staticmethod
    def _render(stem: str, choices: list[str]) -> str:
        lines = [stem, "", "Choices:"]
        lines += [f"({index}) {text}" for index, text in enumerate(choices)]
        return "\n".join(lines)

    def match_label(self, prediction: str, item: Item) -> str:
        """The index the model gave: boxed if it boxed one, else a bare last line."""
        allowed = {p.strip() for p in item.label_space.split(",") if p.strip()}
        boxed = _BOXED_INDEX_RE.findall(prediction)
        if boxed and boxed[-1] in allowed:
            return boxed[-1]
        for line in reversed(prediction.strip().splitlines()):
            token = line.strip().strip("()[].:*\"' ")
            if token in allowed:
                return token
        return TabularAdapter(self.config).match_label(prediction, item)


@register_adapter
class MMLUProWithOptionsAdapter(MMLUProAdapter):
    """MMLU-Pro with the answer options inside the context.

    ``mmlu_pro`` keeps the options out of the context, so a reflexion summary is
    built from the question alone and the options join only at the final answer.
    Here the context holds them, so the summary is built with the options in
    view. Item ids are unchanged.
    """

    name = "mmlu_pro_opts"

    def load(self, path: Path) -> list[Item]:
        return [replace(item, context=item.prompt_text) for item in super().load(path)]

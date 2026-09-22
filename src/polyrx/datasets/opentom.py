"""OpenToM: theory-of-mind questions over short narratives.

Needs its own adapter for one reason the tabular one cannot cover: the allowed
answers depend on the question. "Where will Sam look for the apple?" is scored
against the two places in that story, not against a fixed list.

Several questions share each narrative, so items are sampled by story.
"""

from __future__ import annotations

import json
from pathlib import Path

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter


@register_adapter
class OpenToMAdapter(DatasetAdapter):
    name = "opentom"

    def load(self, path: Path) -> list[Item]:
        entries = json.loads(path.read_text(encoding="utf-8"))
        items: list[Item] = []
        for entry in entries:
            narrative = entry["narrative"]
            question = entry["question"]
            plot = entry.get("plot_info", {})
            items.append(
                Item(
                    item_id=Item.content_id(
                        narrative,
                        question["question"],
                        id_chars=self.config.id_hash_chars,
                    ),
                    context=narrative,
                    prompt_text=narrative,
                    question=question["question"],
                    gold_label=question["answer"],
                    label_space=self.label_space(question["question"], question["type"], plot),
                    group=question["type"],
                    metadata={"plot_info": plot},
                )
            )
        return items

    @staticmethod
    def label_space(question: str, question_type: str, plot: dict) -> str:
        """Allowed answers for one question, from its type and its story.

        Location questions are scored against the two places that story
        mentions, which is why this cannot be a fixed list in config.
        """
        lowered = question.lower()
        if question_type == "attitude":
            return "positive, negative, neutral"
        if "accessible" in lowered:
            return "more accessible, equally accessible, less accessible"
        if "full" in lowered:
            return "more full, equally full, less full"
        if "initial location" in lowered:
            return "Yes, No"

        places: list[str] = []
        for key in ("original_place", "move_to_place"):
            place = plot.get(key, "")
            if place and place not in places:
                places.append(place)
        return ", ".join(places) if places else "Yes, No"

    def match_label(self, prediction: str, item: Item) -> str:
        # The matching rule is the generic one; only the label space is special.
        from polyrx.datasets.tabular import TabularAdapter

        return TabularAdapter(self.config).match_label(prediction, item)

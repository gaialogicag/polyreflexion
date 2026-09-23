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

    #: Closed answer sets, keyed by what the question is about. Taken from the
    #: corpus itself: `attitude` uses three labels and the two `multihop` types
    #: use six between them, across all 13708 questions.
    ATTITUDES = "positive, negative, neutral"
    ACCESSIBILITY = "more accessible, equally accessible, less accessible"
    FULLNESS = "more full, equally full, less full"

    @staticmethod
    def label_space(question: str, question_type: str, plot: dict) -> str:
        """Allowed answers for one question, from its type and its story.

        Driven by ``question_type`` first, because the dataset states it and
        the question wording does not have to. An earlier version decided this
        by looking for the word "accessible" in the question; the questions say
        "accessibility", which does not contain it, so 2386 questions -- every
        accessibility question in the corpus -- were given the story's two
        places as their allowed answers instead. The label space is rendered
        into the answering prompt, so those questions asked the model to choose
        between answers that were all wrong.

        Location questions are still scored against the two places that story
        mentions, which is why this cannot be a fixed list in config.
        """
        lowered = question.lower()
        if question_type == "attitude":
            return OpenToMAdapter.ATTITUDES

        if question_type.startswith("multihop"):
            # Match on the stem, so "accessible" and "accessibility" both land.
            # Neither stem present means the wording changed upstream: offer
            # both sets rather than silently answering the wrong question.
            if "accessib" in lowered:
                return OpenToMAdapter.ACCESSIBILITY
            if "full" in lowered:
                return OpenToMAdapter.FULLNESS
            return f"{OpenToMAdapter.ACCESSIBILITY}, {OpenToMAdapter.FULLNESS}"

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

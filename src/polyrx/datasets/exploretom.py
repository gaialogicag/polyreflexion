"""ExploreToM: program-generated theory-of-mind stories.

Needs its own adapter for the same reason OpenToM does, only more so: the
source states no allowed answers at all. "In which container will Elijah
search for the harmonica?" is scored against the containers that story
mentions, and those differ per story. The allowed answers are therefore
derived from ``story_structure``, the templated form of the story, which names
every room and container in a fixed sentence grammar.

Two columns describe the same story. ``story_structure`` is the generated
template ("Kaylee entered the hotel lobby."); ``infilled_story`` is the same
events written as prose. ``dataset.fields.context`` picks which one the model
reads, and defaults to ``infilled_story``. The label space is always derived
from ``story_structure``, whichever one is shown.

Many questions share each story, so items are sampled by story.
"""

from __future__ import annotations

import ast
import csv
import re
from pathlib import Path

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter

#: Column holding the templated story the label space is derived from.
STRUCTURE_COLUMN = "story_structure"
#: Column shown to the model when the config names none.
DEFAULT_STORY_COLUMN = "infilled_story"
#: Column holding the question-shape tuple, e.g.
#: ``"(None, 'silver letter opener', 'memory-container_location')"``.
PARAMS_COLUMN = "qprop=params"
#: Column holding the order of mental-state nesting: -1 factual, 1 or 2.
ORDER_COLUMN = "qprop=nth_order"

#: Answer sets that do not depend on the story. Taken from the corpus: every
#: knowledge and object-state question in the 13309-row sample uses one of
#: these two, and which one is stated by the question's own wording.
YES_NO = "yes, no"
KNOWS = "knows about it, does not know about it"

_SENTENCE = re.compile(r"(?<=\.)\s+")
# "moved the X to the CONTAINER, which is also located in the ROOM."
_MOVE_TO_CONTAINER = re.compile(
    r"\bmoved the .+? to the (?P<container>.+?), which is (?:also )?located in the (?P<room>.+?)[.,]"
)
# "moved the X to the ROOM[, leaving the CONTAINER in its original location]."
_MOVE_TO_ROOM = re.compile(
    r"\bmoved the .+? to the (?P<room>.+?)"
    r"(?:, leaving the (?P<container>.+?) in its original location)?\."
)
_ENTERED_ROOM = re.compile(r"\b(?:entered|left|exited) the (?P<room>.+?)\.")
# "told privately to Y that the OBJECT is in the CONTAINER."
_OBJECT_IN_CONTAINER = re.compile(r"\bthat the .+? is in the (?P<container>.+?)[.,]")
# "told privately to Y that Z is in the ROOM."
_IS_IN_ROOM = re.compile(r"\bis in the (?P<room>.+?)[.,]")


@register_adapter
class ExploreToMAdapter(DatasetAdapter):
    name = "exploretom"

    def load(self, path: Path) -> list[Item]:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        if not rows:
            raise ValueError(f"No rows in {path}")

        fields = self.config.fields
        story_column = fields.context or DEFAULT_STORY_COLUMN
        if story_column not in rows[0]:
            raise ValueError(
                f"{path} has no column {story_column!r}. Set dataset.fields.context to "
                f"one of the story columns. Columns present: {sorted(rows[0])}"
            )

        minimum = self.config.min_label_space
        items: list[Item] = []
        seen: set[str] = set()
        for row in rows:
            structure = (row.get(STRUCTURE_COLUMN) or "").strip()
            story = (row.get(story_column) or "").strip()
            question = (row.get(fields.question) or "").strip()
            answer = (row.get(fields.answer) or "").strip()
            if not structure or not story or not question or not answer:
                continue

            # Identity is the story and the question, not the answer. The sample
            # holds one pair that appears twice with two different gold answers;
            # keeping both would score a contradiction, so the second is dropped.
            item_id = Item.content_id(story, question, id_chars=self.config.id_hash_chars)
            if item_id in seen:
                continue

            label_space = self.label_space(question, row.get(PARAMS_COLUMN, ""), structure)
            # A question whose allowed answers number one is free marks: the
            # prompt states the answer. Those items measure nothing, so they are
            # dropped rather than counted as correct.
            if minimum and len(_split(label_space)) < minimum:
                continue

            seen.add(item_id)
            items.append(
                Item(
                    item_id=item_id,
                    context=story,
                    prompt_text=story,
                    question=question,
                    gold_label=answer,
                    label_space=label_space,
                    group=self.group(row),
                    metadata={
                        "story_type": row.get("param=story_type", ""),
                        "nth_order": row.get(ORDER_COLUMN, ""),
                        "is_false_belief_story": row.get("sprop=is_false_belief_story_1st", ""),
                    },
                )
            )
        return items

    # -- label space --------------------------------------------------------

    @staticmethod
    def question_kind(params: str) -> str:
        """The question's shape, as the generator recorded it.

        ``qprop=params`` is a Python tuple written as a string; its last element
        names the shape -- ``memory-container_location``, ``room_location-True``,
        ``<knowledge>-False``, ``<state_update> object is signed by the artist``.
        Read from the column rather than from the question's wording, because
        the wording is free to change upstream and the column is generated.
        """
        try:
            parsed = ast.literal_eval(params)
        except (SyntaxError, ValueError):
            return ""
        return str(parsed[-1]) if isinstance(parsed, tuple | list) and parsed else ""

    @classmethod
    def label_space(cls, question: str, params: str, structure: str) -> str:
        """Allowed answers for one question, from its shape and its story."""
        kind = cls.question_kind(params)
        rooms, containers = places(structure)

        if "container_location" in kind:
            return ", ".join(containers)
        if "room_location" in kind:
            return ", ".join(rooms)
        # Knowledge and object-state questions are closed. Which of the two
        # closed sets applies is only in the wording: the generator writes the
        # options into the question itself for the "knows about it" form, and
        # appends "Answer yes or no." for the other.
        if "knows about it" in question.casefold():
            return KNOWS
        return YES_NO

    # -- grouping -----------------------------------------------------------

    @staticmethod
    def group(row: dict[str, str]) -> str:
        """What the question asks about, and how deep the belief nesting goes.

        Reported together because they vary independently: a container-location
        question can be factual, first order or second order, and the gap
        between those is the thing the benchmark exists to measure.
        """
        kind = ExploreToMAdapter.question_kind(row.get(PARAMS_COLUMN, ""))
        if kind.startswith("<knowledge>"):
            family = "knowledge"
        elif kind.startswith("<state_update>"):
            family = "object state"
        elif "container_location" in kind:
            family = "container location"
        elif "room_location" in kind:
            family = "room location"
        else:
            family = "unspecified"

        order = {"-1": "factual", "1": "1st order", "2": "2nd order"}.get(
            str(row.get(ORDER_COLUMN, "")).strip(), "unknown order"
        )
        return f"{family}, {order}"

    # -- scoring ------------------------------------------------------------

    def match_label(self, prediction: str, item: Item) -> str:
        # The matching rule is the generic one; only the label space is special.
        from polyrx.datasets.tabular import TabularAdapter

        return TabularAdapter(self.config).match_label(prediction, item)


def _split(label_space: str) -> list[str]:
    return [part.strip() for part in label_space.split(",") if part.strip()]


def places(structure: str) -> tuple[list[str], list[str]]:
    """Every room and every container the templated story names, in order.

    The generated sentences are a small fixed grammar, so this is a parse and
    not a guess: a room is what someone enters, leaves, or is said to be in; a
    container is what an object is moved into or said to be in. Both are read
    from ``story_structure`` even when the model is shown ``infilled_story``,
    because the prose form paraphrases the names ("the desk drawer" for "the
    wooden desk drawer") and the gold answers use the templated spelling.

    Checked against the whole 13309-row sample: every gold answer is in the
    list this returns.
    """
    rooms: list[str] = []
    containers: list[str] = []

    def add(target: list[str], value: str) -> None:
        value = value.strip()
        if value and value not in target:
            target.append(value)

    for raw in _SENTENCE.split(structure):
        sentence = raw.strip()
        if not sentence:
            continue
        if not sentence.endswith("."):
            sentence += "."

        match = _MOVE_TO_CONTAINER.search(sentence)
        if match:
            add(containers, match.group("container"))
            add(rooms, match.group("room"))
            continue
        match = _OBJECT_IN_CONTAINER.search(sentence)
        if match:
            add(containers, match.group("container"))
            continue
        match = _ENTERED_ROOM.search(sentence)
        if match:
            add(rooms, match.group("room"))
            continue
        match = _MOVE_TO_ROOM.search(sentence)
        if match:
            add(rooms, match.group("room"))
            if match.group("container"):
                add(containers, match.group("container"))
            continue
        match = _IS_IN_ROOM.search(sentence)
        if match:
            add(rooms, match.group("room"))

    # A room reached by "moved the lantern to the control room" also matches the
    # container patterns in other sentences; rooms win, since a story never
    # names the same place both ways.
    containers = [c for c in containers if c not in rooms]
    return rooms, containers

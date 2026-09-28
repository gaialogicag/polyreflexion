"""The config-driven dataset adapter.

Adding a dataset is meant to be configuration only, so most of what could go
wrong here is a field map that silently produces no items, or an answer matcher
that reads a letter out of ordinary prose.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from polyrx.config import DatasetConfig, FieldMap
from polyrx.datasets.base import Item, sample_by_group
from polyrx.datasets.tabular import TabularAdapter


def adapter(**fields: object) -> TabularAdapter:
    config = DatasetConfig(name="test", adapter="tabular", fields=FieldMap(**fields))  # type: ignore[arg-type]
    return TabularAdapter(config)


@pytest.fixture
def write(tmp_path: Path) -> Callable[[str, str], Path]:
    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.write_text(text, encoding="utf-8")
        return path

    return _write


class TestFileFormats:
    def test_csv(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.csv", "q,a\nIs it hidden?,Yes\n")

        items = adapter(question="q", answer="a").load(path)

        assert [(i.question, i.gold_label) for i in items] == [("Is it hidden?", "Yes")]

    def test_tsv_uses_tabs(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.tsv", "q\ta\nIs it hidden?\tYes\n")

        assert adapter(question="q", answer="a").load(path)[0].gold_label == "Yes"

    def test_jsonl(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.jsonl", '{"q": "Is it hidden?", "a": "Yes"}\n\n')

        assert adapter(question="q", answer="a").load(path)[0].question == "Is it hidden?"

    def test_json_array(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.json", '[{"q": "Is it hidden?", "a": "Yes"}]')

        assert len(adapter(question="q", answer="a").load(path)) == 1

    def test_json_object_wrapping_the_rows(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.json", '{"split": "test", "rows": [{"q": "Q?", "a": "Yes"}]}')

        assert len(adapter(question="q", answer="a").load(path)) == 1


class TestFieldMapping:
    def test_context_is_carried_separately_from_the_question(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "passage,q,a\nAnn hid it.,Is it hidden?,Yes\n")

        item = adapter(context="passage", question="q", answer="a").load(path)[0]

        assert item.context == "Ann hid it."
        assert item.question == "Is it hidden?"

    def test_without_a_context_column_the_question_is_the_context(
        self, write: Callable[[str, str], Path]
    ) -> None:
        """The reflexion tree needs something to reason over. For a
        self-contained question that is the question itself."""
        path = write("d.csv", "q,a\nWhy is the sky blue?,Scattering\n")

        assert adapter(question="q", answer="a").load(path)[0].context == "Why is the sky blue?"

    def test_the_group_column_drives_per_group_metrics(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,kind\nQ?,Yes,first-order\n")

        item = adapter(question="q", answer="a", group="kind").load(path)[0]

        assert item.group == "first-order"

    def test_without_a_group_column_items_are_unspecified(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a\nQ?,Yes\n")

        assert adapter(question="q", answer="a").load(path)[0].group == "unspecified"

    def test_an_empty_group_value_falls_back_to_unspecified(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,kind\nQ?,Yes,\n")

        assert adapter(question="q", answer="a", group="kind").load(path)[0].group == "unspecified"

    def test_rows_missing_a_question_or_answer_are_dropped(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a\nQ1?,Yes\n,No\nQ3?,\nQ4?,No\n")

        assert len(adapter(question="q", answer="a").load(path)) == 2

    def test_a_field_map_that_matches_nothing_names_the_real_columns(
        self, write: Callable[[str, str], Path]
    ) -> None:
        """The failure mode of a config-driven adapter: a typo in the field map
        produces zero items rather than an error at the typo."""
        path = write("d.csv", "passage,q,a\nAnn hid it.,Q?,Yes\n")

        with pytest.raises(ValueError) as excinfo:
            adapter(question="question", answer="answer").load(path)

        message = str(excinfo.value)
        assert "'a'" in message and "'q'" in message

    def test_an_empty_file_says_so(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.csv", "q,a\n")

        with pytest.raises(ValueError, match="file is empty"):
            adapter(question="q", answer="a").load(path)


class TestLabelSpace:
    def test_a_per_row_label_space_column(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.csv", 'q,a,allowed\nQ?,Yes,"Yes, No"\n')

        item = adapter(question="q", answer="a", label_space="allowed").load(path)[0]

        assert item.allowed_labels() == ["Yes", "No"]

    def test_a_list_written_as_a_string_is_parsed(self, write: Callable[[str, str], Path]) -> None:
        """CSV sources routinely store a list as its Python repr."""
        path = write("d.csv", "q,a,allowed\nQ?,Yes,\"['Yes', 'No', 'Unknown']\"\n")

        item = adapter(question="q", answer="a", label_space="allowed").load(path)[0]

        assert item.allowed_labels() == ["Yes", "No", "Unknown"]

    def test_a_real_list_in_json_is_used_directly(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.jsonl", '{"q": "Q?", "a": "Yes", "allowed": ["Yes", "No"]}\n')

        item = adapter(question="q", answer="a", label_space="allowed").load(path)[0]

        assert item.allowed_labels() == ["Yes", "No"]

    def test_a_malformed_list_string_is_kept_as_text(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,allowed\nQ?,Yes,\"['Yes', 'No'\"\n")

        item = adapter(question="q", answer="a", label_space="allowed").load(path)[0]

        assert item.label_space == "['Yes', 'No'"

    def test_the_dataset_wide_label_space_is_the_fallback(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a\nQ?,Yes\n")
        config = DatasetConfig(
            name="t", fields=FieldMap(question="q", answer="a"), label_space="Yes, No"
        )

        item = TabularAdapter(config).load(path)[0]

        assert item.allowed_labels() == ["Yes", "No"]


class TestChoiceColumns:
    def test_options_in_their_own_columns_are_lettered(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,o1,o2,o3\nQ?,B,first,second,third\n")

        item = adapter(question="q", answer="a", choices=["o1", "o2", "o3"]).load(path)[0]

        assert item.label_space == "A, B, C"
        assert item.gold_label == "B"

    def test_the_options_are_rendered_into_the_prompt(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,o1,o2\nWhich?,A,first,second\n")

        item = adapter(question="q", answer="a", choices=["o1", "o2"]).load(path)[0]

        assert "(A) first" in item.prompt_text
        assert "(B) second" in item.prompt_text

    def test_a_parenthesised_gold_letter_is_accepted(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,o1,o2\nQ?,(B),first,second\n")

        assert (
            adapter(question="q", answer="a", choices=["o1", "o2"]).load(path)[0].gold_label == "B"
        )

    def test_a_gold_answer_given_as_option_text_resolves_to_its_letter(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,o1,o2\nQ?,second,first,second\n")

        assert (
            adapter(question="q", answer="a", choices=["o1", "o2"]).load(path)[0].gold_label == "B"
        )

    def test_a_gold_answer_matching_no_option_is_an_error(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,a,o1,o2\nQ?,third,first,second\n")

        with pytest.raises(ValueError, match="Cannot identify the correct option"):
            adapter(question="q", answer="a", choices=["o1", "o2"]).load(path)

    def test_blank_option_columns_are_dropped(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.csv", "q,a,o1,o2,o3\nQ?,A,first,second,\n")

        item = adapter(question="q", answer="a", choices=["o1", "o2", "o3"]).load(path)[0]

        assert item.label_space == "A, B"


class TestShuffledChoices:
    def test_the_correct_answer_is_not_always_first(
        self, write: Callable[[str, str], Path]
    ) -> None:
        """A model that always answers (A) must not score above chance."""
        rows = "\n".join(f"Q{i}?,right{i},wrong{i}a,wrong{i}b" for i in range(30))
        path = write("d.csv", "q,correct,w1,w2\n" + rows + "\n")

        items = adapter(
            question="q", answer="correct", correct_choice="correct", incorrect_choices=["w1", "w2"]
        ).load(path)

        assert len({item.gold_label for item in items}) > 1

    def test_the_shuffle_is_stable_across_loads(self, write: Callable[[str, str], Path]) -> None:
        """An order that changed between runs would break every cached summary
        and make two runs unmergeable."""
        path = write("d.csv", "q,correct,w1,w2\nQ?,right,wrong-a,wrong-b\n")
        fields = dict(
            question="q", answer="correct", correct_choice="correct", incorrect_choices=["w1", "w2"]
        )

        first = adapter(**fields).load(path)[0]
        second = adapter(**fields).load(path)[0]

        assert first.prompt_text == second.prompt_text
        assert first.gold_label == second.gold_label

    def test_the_gold_letter_follows_the_shuffled_position(
        self, write: Callable[[str, str], Path]
    ) -> None:
        path = write("d.csv", "q,correct,w1,w2\nQ?,right,wrong-a,wrong-b\n")

        item = adapter(
            question="q", answer="correct", correct_choice="correct", incorrect_choices=["w1", "w2"]
        ).load(path)[0]

        rendered = [ln for ln in item.prompt_text.splitlines() if ln.startswith("(")]
        assert f"({item.gold_label}) right" in rendered


class TestItemIdentity:
    def test_the_id_is_content_addressed_not_positional(
        self, write: Callable[[str, str], Path]
    ) -> None:
        """Re-sampling, merging and extending a run must refer to the same item."""
        first = write("a.csv", "q,a\nQ1?,Yes\nQ2?,No\n")
        second = write("b.csv", "q,a\nQ2?,No\nQ1?,Yes\n")
        load = adapter(question="q", answer="a").load

        assert {i.item_id for i in load(first)} == {i.item_id for i in load(second)}

    def test_different_content_gets_a_different_id(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.csv", "q,a\nQ1?,Yes\nQ2?,Yes\n")

        items = adapter(question="q", answer="a").load(path)

        assert items[0].item_id != items[1].item_id


class TestLoadItems:
    def test_an_item_whose_gold_answer_is_not_allowed_is_rejected(
        self, write: Callable[[str, str], Path]
    ) -> None:
        """The label space goes into the prompt as the allowed answers, so such
        an item asks the model for a string it forbids. No model and no judge
        can score it."""
        path = write("d.csv", 'q,a,allowed\nQ?,Maybe,"Yes, No"\n')

        with pytest.raises(RuntimeError):
            adapter(question="q", answer="a", label_space="allowed").load_items(path)

    def test_consistent_items_load_through(self, write: Callable[[str, str], Path]) -> None:
        path = write("d.csv", 'q,a,allowed\nQ?,Yes,"Yes, No"\n')

        assert len(adapter(question="q", answer="a", label_space="allowed").load_items(path)) == 1


class TestMatchLabel:
    @pytest.fixture
    def yes_no(self) -> Item:
        return Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="Yes",
            label_space="Yes, No",
        )

    @pytest.fixture
    def lettered(self) -> Item:
        return Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="B",
            label_space="A, B, C, D",
        )

    def test_a_bare_label(self, yes_no: Item) -> None:
        assert TabularAdapter(DatasetConfig()).match_label("Yes", yes_no) == "Yes"

    def test_an_answer_prefix_is_stripped(self, yes_no: Item) -> None:
        assert TabularAdapter(DatasetConfig()).match_label("Answer: No", yes_no) == "No"

    def test_matching_ignores_case_and_trailing_marks(self, yes_no: Item) -> None:
        assert TabularAdapter(DatasetConfig()).match_label("yes.", yes_no) == "Yes"

    def test_the_last_line_wins(self, yes_no: Item) -> None:
        """Reasoning comes first, the answer comes last."""
        response = "Ann might think No.\nBut the box was moved.\nYes"

        assert TabularAdapter(DatasetConfig()).match_label(response, yes_no) == "Yes"

    def test_a_longer_label_is_not_swallowed_by_a_shorter_one(self) -> None:
        """A label must not lose to its own suffix. Both name the same span of
        text, so the whole label wins rather than its tail."""
        item = Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="more full",
            label_space="full, more full, less full",
        )

        assert TabularAdapter(DatasetConfig()).match_label("It is more full", item) == "more full"

    def test_the_last_label_mentioned_still_wins(self) -> None:
        """The rule the suffix fix must not break: a model that rules one
        option out and then names another is taken at its final word."""
        item = Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="less full",
            label_space="full, more full, less full",
        )

        matched = TabularAdapter(DatasetConfig()).match_label(
            "It is more full, no -- it is less full", item
        )

        assert matched == "less full"

    def test_a_label_wins_over_a_prefix_of_itself(self) -> None:
        """Prefix containment, the direction that already worked."""
        item = Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="full house",
            label_space="full, full house",
        )

        assert TabularAdapter(DatasetConfig()).match_label("I see a full house", item) == (
            "full house"
        )

    def test_a_shorter_label_still_wins_when_it_genuinely_comes_later(self) -> None:
        """The fix must not turn into "longest always wins": here the two are
        separate occurrences, and the later one is the answer."""
        item = Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="No",
            label_space="No, Not at all",
        )

        assert TabularAdapter(DatasetConfig()).match_label("Not at all sure, so No", item) == "No"

    def test_an_unmatched_answer_comes_back_unchanged(self, yes_no: Item) -> None:
        """The judge decides what to do with it, rather than the matcher
        guessing and scoring a wrong label as right."""
        assert TabularAdapter(DatasetConfig()).match_label("Impossible to say", yes_no) == (
            "Impossible to say"
        )

    def test_no_label_space_returns_the_prediction(self) -> None:
        item = Item(
            item_id="x", context="c", prompt_text="p", question="q", gold_label="42", label_space=""
        )

        assert TabularAdapter(DatasetConfig()).match_label("  42 ", item) == "42"


class TestMatchLetter:
    @pytest.fixture
    def lettered(self) -> Item:
        return Item(
            item_id="x",
            context="c",
            prompt_text="p",
            question="q",
            gold_label="B",
            label_space="A, B, C, D",
        )

    @pytest.mark.parametrize(
        ("response", "expected"),
        [
            pytest.param("B", "B", id="bare-letter"),
            pytest.param("(C)", "C", id="parenthesised"),
            pytest.param("**D**", "D", id="bolded"),
            pytest.param(r"\boxed{A}", "A", id="boxed"),
            pytest.param("Answer: C", "C", id="answer-prefix"),
            pytest.param("the answer is D", "D", id="answer-is"),
            pytest.param("I choose B", "B", id="choose-verb"),
            pytest.param("B. because the box moved", "B", id="letter-then-period"),
        ],
    )
    def test_letters_in_answer_shaped_positions(
        self, response: str, expected: str, lettered: Item
    ) -> None:
        assert TabularAdapter(DatasetConfig()).match_label(response, lettered) == expected

    def test_an_article_is_not_read_as_an_answer(self, lettered: Item) -> None:
        """The regression the position patterns exist for: a word-boundary
        search would score "a cat sat" as answer A."""
        response = "a cat sat on the mat and nothing else happened"

        assert TabularAdapter(DatasetConfig()).match_label(response, lettered) == response

    def test_a_letter_mentioned_in_passing_is_not_taken(self, lettered: Item) -> None:
        response = "Some reasoning that rules things out without committing"

        assert TabularAdapter(DatasetConfig()).match_label(response, lettered) == response


class TestSampleByGroup:
    def items(self, count: int, *, per_context: int = 1) -> list[Item]:
        out = []
        for c in range(count):
            for q in range(per_context):
                out.append(
                    Item(
                        item_id=f"i{c}-{q}",
                        context=f"passage {c}",
                        prompt_text="p",
                        question=f"q{q}",
                        gold_label="Yes",
                        label_space="Yes, No",
                    )
                )
        return out

    def test_questions_sharing_a_passage_are_sampled_together(self) -> None:
        """Sampling by question would put one passage under several conditions
        and build its summary more than once."""
        sampled = sample_by_group(self.items(10, per_context=3), count=2, seed=0)

        assert len(sampled) == 6
        assert len({item.context for item in sampled}) == 2

    def test_item_grouping_treats_each_question_as_its_own_group(self) -> None:
        sampled = sample_by_group(self.items(10, per_context=3), count=2, seed=0, group_key="item")

        assert len(sampled) == 2

    def test_the_same_seed_gives_the_same_sample(self) -> None:
        items = self.items(20)

        first = sample_by_group(items, count=5, seed=7)
        second = sample_by_group(items, count=5, seed=7)

        assert [i.item_id for i in first] == [i.item_id for i in second]

    def test_a_different_seed_gives_a_different_sample(self) -> None:
        items = self.items(50)

        first = {i.item_id for i in sample_by_group(items, count=5, seed=1)}
        second = {i.item_id for i in sample_by_group(items, count=5, seed=2)}

        assert first != second

    def test_asking_for_more_than_exists_returns_everything(self) -> None:
        sampled = sample_by_group(self.items(3), count=99, seed=0)

        assert len(sampled) == 3

    def test_excluded_items_are_not_sampled(self) -> None:
        items = self.items(10)

        sampled = sample_by_group(items, count=10, seed=0, exclude={"i0-0", "i1-0"})

        assert {"i0-0", "i1-0"}.isdisjoint({i.item_id for i in sampled})

    def test_only_selects_exactly_those_items(self) -> None:
        items = self.items(10)

        sampled = sample_by_group(items, count=2, seed=0, only={"i3-0", "i7-0"})

        assert {i.item_id for i in sampled} == {"i3-0", "i7-0"}

"""Token and cost accounting.

Tokens are a durable fact about a run; money is those tokens put through a
price table that changes without notice and does not exist for a local model.
The distinction matters most in one place: a condition nothing could price has
a cost of ``None``, never ``0.0``, because a zero reads as "this method was
free".
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pytest

from polyrx import usage as usage_module
from polyrx.usage import (
    UNATTRIBUTED,
    CallCost,
    Pricer,
    UsageRecorder,
    active_recorder,
    attributed_to,
    build_pricer,
    record_call,
    set_active_recorder,
    submit_in_context,
    totals,
)


def flat_pricer(amount: float = 0.01, source: str = "test-table") -> Pricer:
    """Prices every call the same, so assertions are about the arithmetic."""

    def price(*, model: str, provider: str, **_: int) -> CallCost | None:
        return CallCost(amount, source)

    return price


def selective_pricer(priced_model: str) -> Pricer:
    """Prices one model and gives up on the rest, like a real table meeting a
    local model or a name it has never seen."""

    def price(*, model: str, provider: str, **_: int) -> CallCost | None:
        return CallCost(0.01, "test-table") if model == priced_model else None

    return price


@pytest.fixture
def restore_active_recorder() -> object:
    saved = usage_module._active
    yield
    usage_module._active = saved


class TestTokenAccumulation:
    def test_calls_to_the_same_model_accumulate(self) -> None:
        recorder = UsageRecorder()
        for _ in range(3):
            recorder.record(
                model="gpt-4o", provider="openai", prompt_tokens=100, completion_tokens=20
            )

        entry = recorder.snapshot()[UNATTRIBUTED]
        assert entry["calls"] == 3
        assert entry["prompt_tokens"] == 300
        assert entry["completion_tokens"] == 60

    def test_max_prompt_tokens_is_the_largest_call_not_the_sum(self) -> None:
        """It exists to show which price band a run actually sat in. A sum
        would claim a band no single request ever reached."""
        recorder = UsageRecorder()
        for size in (100, 900, 400):
            recorder.record(
                model="gpt-4o", provider="openai", prompt_tokens=size, completion_tokens=1
            )

        assert recorder.snapshot()[UNATTRIBUTED]["max_prompt_tokens"] == 900

    def test_cached_and_reasoning_tokens_are_recorded_apart(self) -> None:
        recorder = UsageRecorder()
        recorder.record(
            model="gpt-4o",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=80,
            reasoning_tokens=40,
        )

        entry = recorder.snapshot()[UNATTRIBUTED]
        assert entry["cached_tokens"] == 80
        assert entry["reasoning_tokens"] == 40

    def test_cached_and_reasoning_are_parts_of_the_totals_not_additions(self) -> None:
        """Documents the contract the clients rely on: adding them again would
        double-count every cached prompt."""
        recorder = UsageRecorder()
        recorder.record(
            model="gpt-4o",
            provider="openai",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=80,
            reasoning_tokens=40,
        )

        entry = recorder.snapshot()[UNATTRIBUTED]
        assert entry["prompt_tokens"] == 100
        assert entry["completion_tokens"] == 50

    def test_none_token_counts_are_treated_as_zero(self) -> None:
        """A provider that omits usage returns None, not 0."""
        recorder = UsageRecorder()
        recorder.record(
            model="gpt-4o",
            provider="openai",
            prompt_tokens=None,  # type: ignore[arg-type]
            completion_tokens=None,  # type: ignore[arg-type]
        )

        assert recorder.snapshot()[UNATTRIBUTED]["prompt_tokens"] == 0

    def test_models_are_tallied_separately_within_a_condition(self) -> None:
        recorder = UsageRecorder()
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)
        recorder.record(
            model="gpt-5-mini", provider="openai", prompt_tokens=20, completion_tokens=2
        )

        entry = recorder.snapshot()[UNATTRIBUTED]
        assert set(entry["by_model"]) == {"gpt-4o", "gpt-5-mini"}
        assert entry["prompt_tokens"] == 30


class TestAttribution:
    def test_calls_land_under_the_condition_in_scope(self) -> None:
        recorder = UsageRecorder()
        with attributed_to("answerer_meta_c2"):
            recorder.record(
                model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1
            )

        assert set(recorder.snapshot()) == {"answerer_meta_c2"}

    def test_calls_outside_any_condition_are_not_added_to_one(self) -> None:
        """The doctor's probe and the interactive tools must not inflate a
        method's cost."""
        recorder = UsageRecorder()
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)

        assert set(recorder.snapshot()) == {UNATTRIBUTED}

    def test_attribution_is_restored_afterwards(self) -> None:
        recorder = UsageRecorder()
        with attributed_to("answerer_d1"):
            pass
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)

        assert set(recorder.snapshot()) == {UNATTRIBUTED}

    def test_nesting_restores_the_outer_condition(self) -> None:
        recorder = UsageRecorder()
        with attributed_to("outer"):
            with attributed_to("inner"):
                recorder.record(model="m", provider="openai", prompt_tokens=1, completion_tokens=1)
            recorder.record(model="m", provider="openai", prompt_tokens=1, completion_tokens=1)

        assert set(recorder.snapshot()) == {"outer", "inner"}

    def test_an_explicit_condition_overrides_the_one_in_scope(self) -> None:
        recorder = UsageRecorder()
        with attributed_to("answerer_d1"):
            recorder.record(
                model="gpt-4o",
                provider="openai",
                prompt_tokens=10,
                completion_tokens=1,
                condition="somewhere_else",
            )

        assert set(recorder.snapshot()) == {"somewhere_else"}


class TestAttributionAcrossThreads:
    def test_a_worker_thread_inherits_the_caller_attribution(self) -> None:
        """A pool thread starts with an empty context. Without the copy, the
        engine's fan-out -- most of the work in a run -- attributes nothing."""
        recorder = UsageRecorder()

        def work() -> None:
            recorder.record(
                model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1
            )

        with ThreadPoolExecutor(max_workers=2) as pool, attributed_to("answerer_d3"):
            submit_in_context(pool, work).result()

        assert set(recorder.snapshot()) == {"answerer_d3"}

    def test_a_plain_submit_loses_the_attribution(self) -> None:
        """The regression ``submit_in_context`` exists to prevent."""
        recorder = UsageRecorder()

        def work() -> None:
            recorder.record(
                model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1
            )

        with ThreadPoolExecutor(max_workers=2) as pool, attributed_to("answerer_d3"):
            pool.submit(work).result()

        assert set(recorder.snapshot()) == {UNATTRIBUTED}

    def test_concurrent_records_do_not_lose_counts(self) -> None:
        recorder = UsageRecorder()

        def work() -> None:
            recorder.record(model="gpt-4o", provider="openai", prompt_tokens=1, completion_tokens=1)

        with ThreadPoolExecutor(max_workers=8) as pool:
            for future in [pool.submit(work) for _ in range(200)]:
                future.result()

        assert recorder.snapshot()[UNATTRIBUTED]["calls"] == 200


class TestPricing:
    def test_without_a_pricer_nothing_is_priced(self) -> None:
        recorder = UsageRecorder()
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)

        entry = recorder.snapshot()[UNATTRIBUTED]
        assert entry["cost"] is None
        assert entry["unpriced_models"] == ["gpt-4o"]

    def test_costs_are_summed_per_call(self) -> None:
        """Each call is priced at its own size, because providers charge a
        higher rate above a prompt-size threshold."""
        recorder = UsageRecorder(flat_pricer(0.01))
        for _ in range(3):
            recorder.record(
                model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1
            )

        assert recorder.snapshot()[UNATTRIBUTED]["cost"] == pytest.approx(0.03)

    def test_the_price_source_is_recorded(self) -> None:
        recorder = UsageRecorder(flat_pricer(0.01, "test-table"))
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)

        assert recorder.snapshot()[UNATTRIBUTED]["cost_sources"] == ["test-table"]

    def test_an_unpriced_model_is_named_beside_a_priced_one(self) -> None:
        """A partial total must say it is partial rather than understate."""
        recorder = UsageRecorder(selective_pricer("gpt-4o"))
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)
        recorder.record(model="phi4-mini", provider="ollama", prompt_tokens=10, completion_tokens=1)

        entry = recorder.snapshot()[UNATTRIBUTED]
        assert entry["cost"] == pytest.approx(0.01)
        assert entry["unpriced_models"] == ["phi4-mini"]

    def test_a_fully_unpriced_condition_costs_none_not_zero(self) -> None:
        """The distinction the module exists for: a local model is free, an
        unknown one is unknown, and a zero reads as free."""
        recorder = UsageRecorder(selective_pricer("gpt-4o"))
        recorder.record(model="phi4-mini", provider="ollama", prompt_tokens=10, completion_tokens=1)

        assert recorder.snapshot()[UNATTRIBUTED]["cost"] is None


class TestBuildPricer:
    def test_returns_none_when_the_price_table_is_not_installed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``cost`` is an optional extra. Without it a run still counts every
        token."""

        def unavailable() -> None:
            raise ImportError("no genai_prices")

        monkeypatch.setattr(usage_module, "GenAIPricer", unavailable)

        assert build_pricer() is None

    def test_returns_a_pricer_when_the_table_is_available(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = object()
        monkeypatch.setattr(usage_module, "GenAIPricer", lambda: sentinel)

        assert build_pricer() is sentinel


class TestSnapshotIsolation:
    def test_a_snapshot_does_not_change_when_more_calls_arrive(self) -> None:
        recorder = UsageRecorder()
        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)
        taken = recorder.snapshot()

        recorder.record(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)

        assert taken[UNATTRIBUTED]["calls"] == 1


class TestTotals:
    def test_sums_every_condition(self) -> None:
        recorder = UsageRecorder(flat_pricer(0.01))
        for condition in ("answerer_direct", "answerer_d1"):
            with attributed_to(condition):
                recorder.record(
                    model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=2
                )

        summed = totals(recorder.snapshot())
        assert summed.calls == 2
        assert summed.prompt_tokens == 20
        assert summed.cost == pytest.approx(0.02)

    def test_cost_is_none_when_nothing_could_be_priced(self) -> None:
        recorder = UsageRecorder()
        recorder.record(model="phi4-mini", provider="ollama", prompt_tokens=10, completion_tokens=1)

        assert totals(recorder.snapshot()).cost is None

    def test_unpriced_models_are_collected_without_duplicates(self) -> None:
        recorder = UsageRecorder()
        for condition in ("a", "b"):
            with attributed_to(condition):
                recorder.record(
                    model="phi4-mini", provider="ollama", prompt_tokens=1, completion_tokens=1
                )

        assert totals(recorder.snapshot()).unpriced_models == ["phi4-mini"]

    def test_an_empty_tally_totals_to_nothing(self) -> None:
        summed = totals({})

        assert summed.calls == 0
        assert summed.cost is None

    def test_the_currency_is_carried_through(self) -> None:
        assert totals({}, currency="EUR").currency == "EUR"


class TestProcessWideRecorder:
    def test_record_call_writes_to_the_installed_recorder(
        self, restore_active_recorder: None
    ) -> None:
        recorder = set_active_recorder(UsageRecorder())
        record_call(model="gpt-4o", provider="openai", prompt_tokens=10, completion_tokens=1)

        assert active_recorder() is recorder
        assert recorder.snapshot()[UNATTRIBUTED]["calls"] == 1

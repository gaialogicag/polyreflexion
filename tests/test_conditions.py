"""Experimental conditions and the grid lock.

A condition's ``summary_key`` is the cache key every summary on disk is stored
under. Changing one does not crash anything -- it orphans the cache silently,
and a run then reports numbers produced at a different depth. That is what the
lock exists to catch, and most of what is tested here.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from polyrx.conditions import (
    Condition,
    ConditionRegistry,
    GridLockError,
    check_against_lock,
    registry_from_config,
    require_lock_match,
    write_lock,
)


class TestValidation:
    def test_negative_depth_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="depth must be >= 0"):
            Condition(name="bad", backend="answerer", depth=-1)

    def test_negative_meta_budget_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="meta_cycles must be >= 0"):
            Condition(name="bad", backend="answerer", depth=1, meta_cycles=-1)

    def test_a_meta_condition_needs_a_reflexion_pass(self) -> None:
        """Cycle 0 of a meta run is a depth-1 reflexion. Depth 0 would mean the
        meta layer ran over nothing."""
        with pytest.raises(ValueError, match="depth must be >= 1"):
            Condition(name="bad", backend="answerer", depth=0, meta_cycles=2)

    def test_the_error_names_the_condition(self) -> None:
        with pytest.raises(ValueError, match="my_condition"):
            Condition(name="my_condition", backend="answerer", depth=-1)


class TestKind:
    def test_direct_is_no_reflexion_and_no_meta(self) -> None:
        condition = Condition(name="d", backend="answerer")

        assert condition.is_direct
        assert not condition.is_reflexion
        assert not condition.is_meta
        assert not condition.uses_summary

    def test_reflexion_is_a_tree_with_no_meta_layer(self) -> None:
        condition = Condition(name="r", backend="answerer", depth=2)

        assert condition.is_reflexion
        assert not condition.is_direct
        assert not condition.is_meta
        assert condition.uses_summary

    def test_meta_is_not_counted_as_plain_reflexion(self) -> None:
        condition = Condition(name="m", backend="answerer", depth=1, meta_cycles=2)

        assert condition.is_meta
        assert not condition.is_reflexion
        assert not condition.is_direct
        assert condition.uses_summary


class TestSummaryKey:
    def test_a_direct_condition_has_no_cache_key(self) -> None:
        """It produces no summary, so there is nothing to cache."""
        assert Condition(name="d", backend="answerer").summary_key == ""

    def test_depth_one_is_keyed_by_the_backend_alone(self) -> None:
        assert Condition(name="r", backend="answerer", depth=1).summary_key == "answerer"

    def test_deeper_trees_carry_the_depth(self) -> None:
        assert Condition(name="r3", backend="answerer", depth=3).summary_key == "answerer_d3"

    def test_meta_is_keyed_by_the_budget(self) -> None:
        condition = Condition(name="m", backend="answerer", depth=1, meta_cycles=2)

        assert condition.summary_key == "answerer_meta_c2"

    @pytest.mark.parametrize(
        "condition",
        [
            Condition(name="direct", backend="answerer"),
            Condition(name="d1", backend="answerer", depth=1),
            Condition(name="d3", backend="answerer", depth=3),
            Condition(name="m1", backend="answerer", depth=1, meta_cycles=1),
            Condition(name="m2", backend="answerer", depth=1, meta_cycles=2),
        ],
    )
    def test_no_two_settings_share_a_key(self, condition: Condition) -> None:
        """The property the cache depends on. If two settings collided, a
        depth-3 number would silently be served from a depth-1 run."""
        others = [
            Condition(name="direct", backend="answerer"),
            Condition(name="d1", backend="answerer", depth=1),
            Condition(name="d3", backend="answerer", depth=3),
            Condition(name="m1", backend="answerer", depth=1, meta_cycles=1),
            Condition(name="m2", backend="answerer", depth=1, meta_cycles=2),
        ]
        clashes = [
            o for o in others if o.name != condition.name and o.summary_key == condition.summary_key
        ]
        assert clashes == []

    def test_the_backend_separates_otherwise_identical_settings(self) -> None:
        hosted = Condition(name="a", backend="answerer", depth=1)
        local = Condition(name="b", backend="open_weights", depth=1)

        assert hosted.summary_key != local.summary_key


class TestPriorSummaryKey:
    def test_budget_two_continues_from_budget_one(self) -> None:
        condition = Condition(name="m2", backend="answerer", depth=1, meta_cycles=2)

        assert condition.prior_summary_key == "answerer_meta_c1"

    def test_budget_one_has_no_predecessor(self) -> None:
        """Its cycle 0 is a fresh reflexion, not a continuation."""
        condition = Condition(name="m1", backend="answerer", depth=1, meta_cycles=1)

        assert condition.prior_summary_key is None

    def test_a_non_meta_condition_has_no_predecessor(self) -> None:
        assert Condition(name="r", backend="answerer", depth=2).prior_summary_key is None


class TestLabel:
    def test_direct(self) -> None:
        assert Condition(name="d", backend="answerer").label() == "answerer, no reflexion"

    def test_reflexion_names_the_depth(self) -> None:
        label = Condition(name="r", backend="answerer", depth=2).label()

        assert label == "answerer, reflexion depth 2"

    def test_one_cycle_is_singular(self) -> None:
        label = Condition(name="m", backend="answerer", depth=1, meta_cycles=1).label()

        assert label.endswith("1 meta cycle")

    def test_several_cycles_are_plural(self) -> None:
        label = Condition(name="m", backend="answerer", depth=1, meta_cycles=3).label()

        assert label.endswith("3 meta cycles")


class TestRegistry:
    @pytest.fixture
    def registry(self) -> ConditionRegistry:
        return ConditionRegistry(
            [
                Condition(name="answerer_direct", backend="answerer"),
                Condition(name="answerer_reflexion", backend="answerer", depth=1),
                Condition(name="answerer_d3", backend="answerer", depth=3),
                Condition(name="open_weights_direct", backend="open_weights"),
            ]
        )

    def test_registration_order_is_preserved(self, registry: ConditionRegistry) -> None:
        """It is also the report column order. Sorting by name would put
        ``answerer_meta_c10`` between ``c1`` and ``c2``."""
        assert registry.names() == (
            "answerer_direct",
            "answerer_reflexion",
            "answerer_d3",
            "open_weights_direct",
        )

    def test_unknown_name_lists_what_is_known(self, registry: ConditionRegistry) -> None:
        with pytest.raises(KeyError) as excinfo:
            registry.get("answerer_meta_c9")

        message = str(excinfo.value)
        assert "answerer_meta_c9" in message
        assert "answerer_direct" in message

    def test_registering_the_same_name_replaces_it(self, registry: ConditionRegistry) -> None:
        registry.register(Condition(name="answerer_direct", backend="open_weights"))

        assert registry.get("answerer_direct").backend == "open_weights"
        assert len(registry.names()) == 4

    def test_resolve_all_follows_registry_order_not_argument_order(
        self, registry: ConditionRegistry
    ) -> None:
        resolved = registry.resolve_all(["answerer_d3", "answerer_direct"])

        assert [c.name for c in resolved] == ["answerer_direct", "answerer_d3"]

    def test_resolve_all_deduplicates(self, registry: ConditionRegistry) -> None:
        resolved = registry.resolve_all(["answerer_direct", "answerer_direct"])

        assert len(resolved) == 1


class TestAliases:
    @pytest.fixture
    def registry(self) -> ConditionRegistry:
        return ConditionRegistry(
            [
                Condition(name="answerer_reflexion", backend="answerer", depth=1),
                Condition(
                    name="gpt_reflexion",
                    backend="answerer",
                    alias_of="answerer_reflexion",
                    note="retired name",
                ),
            ]
        )

    def test_an_alias_behaves_like_its_target(self, registry: ConditionRegistry) -> None:
        alias = registry.get("gpt_reflexion")

        assert alias.depth == 1
        assert alias.summary_key == "answerer"

    def test_an_alias_keeps_its_own_name_and_note(self, registry: ConditionRegistry) -> None:
        """Reports show what the user asked for; only the behaviour is borrowed."""
        alias = registry.get("gpt_reflexion")

        assert alias.name == "gpt_reflexion"
        assert alias.note == "retired name"

    def test_an_alias_shares_the_target_cache_entry(self, registry: ConditionRegistry) -> None:
        """The point of an alias: the retired name must not re-run anything."""
        assert (
            registry.get("gpt_reflexion").summary_key
            == registry.get("answerer_reflexion").summary_key
        )


class TestExpandGroup:
    @pytest.fixture
    def registry(self) -> ConditionRegistry:
        return ConditionRegistry(
            [
                Condition(name="answerer_direct", backend="answerer"),
                Condition(name="answerer_d1", backend="answerer", depth=1),
                Condition(name="answerer_d3", backend="answerer", depth=3),
                Condition(name="open_weights_direct", backend="open_weights"),
            ]
        )

    def test_all_returns_every_condition(self, registry: ConditionRegistry) -> None:
        assert set(registry.expand_group("all")) == set(registry.names())

    def test_a_backend_name_selects_that_backend(self, registry: ConditionRegistry) -> None:
        assert registry.expand_group("open_weights") == ("open_weights_direct",)

    def test_the_nod3_suffix_drops_the_expensive_depth_three_run(
        self, registry: ConditionRegistry
    ) -> None:
        expanded = registry.expand_group("answerer_nod3")

        assert "answerer_d3" not in expanded
        assert set(expanded) == {"answerer_direct", "answerer_d1"}

    def test_an_unknown_group_expands_to_nothing(self, registry: ConditionRegistry) -> None:
        assert registry.expand_group("nonexistent") == ()


class TestBuildingFromConfig:
    def test_accepts_mappings(self) -> None:
        registry = registry_from_config([{"name": "a", "backend": "answerer", "depth": 2}])

        assert registry.get("a").depth == 2

    def test_accepts_objects_with_attributes(self) -> None:
        entry = SimpleNamespace(name="a", backend="answerer", depth=1, meta_cycles=0)

        assert registry_from_config([entry]).get("a").depth == 1

    def test_passes_condition_instances_through(self) -> None:
        condition = Condition(name="a", backend="answerer", depth=3)

        assert registry_from_config([condition]).get("a") is condition

    def test_missing_fields_default_rather_than_fail(self) -> None:
        registry = registry_from_config([{"name": "a", "backend": "answerer"}])
        condition = registry.get("a")

        assert (condition.depth, condition.meta_cycles, condition.note) == (0, 0, "")

    def test_none_is_treated_as_the_default(self) -> None:
        """A YAML key written with no value composes as None, not as missing."""
        registry = registry_from_config(
            [{"name": "a", "backend": "answerer", "depth": None, "alias_of": None}]
        )

        assert registry.get("a").depth == 0
        assert registry.get("a").alias_of is None

    @pytest.mark.parametrize(
        "entry",
        [
            pytest.param({"backend": "answerer"}, id="no-name"),
            pytest.param({"name": "a"}, id="no-backend"),
            pytest.param({}, id="neither"),
        ],
    )
    def test_a_condition_needs_a_name_and_a_backend(self, entry: dict) -> None:
        with pytest.raises(ValueError, match="conf/conditions"):
            registry_from_config([entry])


class TestLock:
    @pytest.fixture
    def registry(self) -> ConditionRegistry:
        return ConditionRegistry(
            [
                Condition(name="answerer_direct", backend="answerer"),
                Condition(name="answerer_d3", backend="answerer", depth=3),
            ]
        )

    def test_a_grid_matches_the_lock_it_was_written_from(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "conditions.lock.json")

        assert check_against_lock(registry, lock) == []

    def test_writing_creates_missing_parent_directories(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "data" / "conditions.lock.json")

        assert lock.is_file()

    def test_the_lock_records_the_cache_key(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "conditions.lock.json")
        payload = json.loads(lock.read_text())

        assert payload["conditions"]["answerer_d3"]["summary_key"] == "answerer_d3"

    def test_a_changed_depth_is_reported_with_both_values(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "conditions.lock.json")
        registry.register(Condition(name="answerer_d3", backend="answerer", depth=2))

        problems = check_against_lock(registry, lock)

        assert any("answerer_d3.depth" in p and "3" in p and "2" in p for p in problems)

    def test_a_changed_depth_also_reports_the_orphaned_cache_key(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        """The cache key is the damage. The depth is only how it happened."""
        lock = write_lock(registry, tmp_path / "conditions.lock.json")
        registry.register(Condition(name="answerer_d3", backend="answerer", depth=2))

        assert any("answerer_d3.summary_key" in p for p in check_against_lock(registry, lock))

    def test_a_condition_removed_from_the_grid_is_reported(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "conditions.lock.json")
        shrunk = ConditionRegistry([Condition(name="answerer_direct", backend="answerer")])

        problems = check_against_lock(shrunk, lock)

        assert problems == ["answerer_d3: in the lock but no longer in the grid"]

    def test_a_condition_added_to_the_grid_is_not_reported(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        """Adding a condition orphans no cache entry, so it is not drift.
        Documents the asymmetry: the check runs lock -> grid only."""
        lock = write_lock(registry, tmp_path / "conditions.lock.json")
        registry.register(Condition(name="answerer_d5", backend="answerer", depth=5))

        assert check_against_lock(registry, lock) == []

    def test_a_missing_lock_is_not_drift(self, registry: ConditionRegistry, tmp_path: Path) -> None:
        """A fresh checkout has no published results to protect yet."""
        assert check_against_lock(registry, tmp_path / "absent.json") == []


class TestRequireLockMatch:
    @pytest.fixture
    def registry(self) -> ConditionRegistry:
        return ConditionRegistry([Condition(name="answerer_d3", backend="answerer", depth=3)])

    def test_a_matching_grid_passes_silently(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "conditions.lock.json")

        require_lock_match(registry, lock)

    def test_drift_raises_and_says_how_to_re_lock(
        self, registry: ConditionRegistry, tmp_path: Path
    ) -> None:
        lock = write_lock(registry, tmp_path / "conditions.lock.json")
        registry.register(Condition(name="answerer_d3", backend="answerer", depth=1))

        with pytest.raises(GridLockError) as excinfo:
            require_lock_match(registry, lock)

        assert "polyrx-conditions lock" in str(excinfo.value)

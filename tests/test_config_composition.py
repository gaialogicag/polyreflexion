"""Every shipped config group must compose.

Hydra resolves the tree at run time, so a config that no longer composes is
found by running the thing, not by importing it. These tests compose each
group the repository ships, which is the cheapest guard against a rename in
one file breaking a defaults list in another.

They compose the tree that ``find_conf_dir`` resolves, so in a checkout they
check the repository's ``conf/`` and in an installed environment they check the
packaged copy.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from polyrx.cli.compose import load_config
from polyrx.resources import find_conf_dir


def config_names(group: str) -> list[str]:
    """The selectable names in one config group, excluding schema bases."""
    directory = find_conf_dir() / group
    if not directory.is_dir():
        return []
    return sorted(
        path.stem for path in directory.glob("*.yaml") if not path.stem.startswith(("base_", "_"))
    )


class TestTheDefaultConfig:
    def test_composes(self) -> None:
        assert load_config([]) is not None

    def test_carries_the_groups_a_run_needs(self) -> None:
        cfg = load_config([])

        for group in ("backends", "dataset", "paths", "prompts", "engine", "conditions"):
            assert getattr(cfg, group, None) is not None, f"missing {group}"

    def test_paths_are_relative_to_the_launch_directory(self) -> None:
        """An installed user runs from their own directory, and results must
        land there rather than beside the package."""
        cfg = load_config([])

        assert not Path(cfg.paths.root).is_absolute() or cfg.paths.root == "."


class TestEveryGroupComposes:
    @pytest.mark.parametrize("name", config_names("experiment"))
    def test_experiment(self, name: str) -> None:
        assert load_config([f"experiment={name}"]) is not None

    @pytest.mark.parametrize("name", config_names("dataset"))
    def test_dataset(self, name: str) -> None:
        cfg = load_config([f"dataset={name}"])

        assert cfg.dataset.name
        assert cfg.dataset.adapter

    @pytest.mark.parametrize("name", config_names("conditions"))
    def test_conditions(self, name: str) -> None:
        assert load_config([f"conditions={name}"]) is not None

    @pytest.mark.parametrize("name", config_names("backends"))
    def test_backends(self, name: str) -> None:
        assert load_config([f"backends={name}"]) is not None

    @pytest.mark.parametrize("name", config_names("prompts/meta"))
    def test_meta_prompt_set(self, name: str) -> None:
        assert load_config([f"prompts/meta={name}"]) is not None


class TestGroupsAreNonEmpty:
    """A glob that silently matches nothing would make the tests above pass by
    running zero cases."""

    @pytest.mark.parametrize(
        "group", ["experiment", "dataset", "conditions", "backends", "prompts/meta"]
    )
    def test_the_group_ships_at_least_one_config(self, group: str) -> None:
        assert config_names(group)


class TestConditionGrids:
    @pytest.mark.parametrize("name", config_names("conditions"))
    def test_every_condition_builds(self, name: str) -> None:
        """A grid entry missing a backend, or with a meta budget on a depth-0
        condition, fails here rather than partway through a paid run."""
        from polyrx.conditions import registry_from_config

        cfg = load_config([f"conditions={name}"])
        registry = registry_from_config(cfg.conditions.conditions)

        assert registry.names()

    @pytest.mark.parametrize("name", config_names("conditions"))
    def test_no_two_distinct_conditions_share_a_cache_key(self, name: str) -> None:
        """Two real conditions on one key means one method's numbers are served
        from the other's cache. An alias is excluded: sharing its target's key
        is exactly what an alias is for, so a retired name re-runs nothing."""
        from polyrx.conditions import registry_from_config

        cfg = load_config([f"conditions={name}"])
        entries = cfg.conditions.conditions
        aliases = {e.name for e in entries if getattr(e, "alias_of", None)} or {
            e["name"] for e in entries if isinstance(e, dict) and e.get("alias_of")
        }
        registry = registry_from_config(entries)

        keys = [
            c.summary_key for c in registry.ordered() if c.summary_key and c.name not in aliases
        ]
        assert len(keys) == len(set(keys))

    @pytest.mark.parametrize("name", config_names("conditions"))
    def test_an_alias_shares_its_target_cache_key(self, name: str) -> None:
        from polyrx.conditions import registry_from_config

        cfg = load_config([f"conditions={name}"])
        entries = cfg.conditions.conditions
        registry = registry_from_config(entries)

        for entry in entries:
            target = getattr(entry, "alias_of", None)
            if target:
                assert registry.get(entry.name).summary_key == registry.get(target).summary_key


class TestOverrides:
    def test_a_prompt_can_be_overridden_on_the_command_line(self) -> None:
        cfg = load_config(['prompts.meta.drift_check="CUSTOM {answer}"'])

        assert cfg.prompts.meta.drift_check == "CUSTOM {answer}"

    def test_an_unknown_group_value_is_an_error(self) -> None:
        from hydra.errors import MissingConfigException

        with pytest.raises(MissingConfigException):
            load_config(["dataset=does_not_exist"])

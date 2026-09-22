"""Register the structured configs so Hydra type-checks the YAML tree.

Without this, a typo in a config file (``max_worker`` for ``max_workers``, a
string where an int belongs) is only discovered when the run reaches the line
that reads it — often after the dataset download and the first API calls.  With
it, composition fails immediately and names the offending key.
"""

from __future__ import annotations

from hydra.core.config_store import ConfigStore

from polyreflexion.config import (
    BackendsConfig,
    DatasetConfig,
    EngineConfig,
    ExperimentConfig,
    MetaLayerConfig,
    MetaPrompts,
    PathsConfig,
    ReflexionPrompts,
    RootConfig,
)


def register() -> ConfigStore:
    """Register every config group. Safe to call more than once."""
    cs = ConfigStore.instance()
    cs.store(name="base_config", node=RootConfig)
    cs.store(group="experiment", name="base_experiment", node=ExperimentConfig)
    cs.store(group="backends", name="base_backends", node=BackendsConfig)
    cs.store(group="dataset", name="base_dataset", node=DatasetConfig)
    cs.store(group="engine", name="base_engine", node=EngineConfig)
    cs.store(group="meta", name="base_meta", node=MetaLayerConfig)
    cs.store(group="paths", name="base_paths", node=PathsConfig)
    # Prompt sets are nested groups, so their schemas are registered under the
    # same group paths the YAML files live in.
    cs.store(group="prompts/reflexion", name="base_reflexion_prompts", node=ReflexionPrompts)
    cs.store(group="prompts/meta", name="base_meta_prompts", node=MetaPrompts)
    return cs

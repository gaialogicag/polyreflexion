"""Compose the Hydra config tree from a plain argparse tool.

Tools that take free text on the command line — ``polyrx ask "why ..."`` — read
badly as Hydra applications, where every argument is a ``key=value`` override.
They still need the same backends, prompts and paths as the benchmarks, so they
compose the config through the API instead of the decorator.
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from polyreflexion.conf_store import register
from polyreflexion.config import RootConfig
from polyreflexion.resources import CONF_DIR


def load_config(overrides: list[str] | None = None) -> RootConfig:
    """Return the composed, type-checked config.

    Parameters
    ----------
    overrides:
        Hydra-style overrides, e.g. ``["meta=ask", "backends.nano.model=gpt-4o"]``.
    """
    register()
    with initialize_config_dir(version_base="1.3", config_dir=str(CONF_DIR)):
        cfg = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]

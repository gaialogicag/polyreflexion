"""Compose the Hydra config tree from a plain argparse tool.

Tools that take free text on the command line — ``polyrx ask "why ..."`` — read
badly as Hydra applications, where every argument is a ``key=value`` override.
They still need the same backends, prompts and paths as the benchmarks, so they
compose the config through the API instead of the decorator.
"""

from __future__ import annotations

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from polyrx.conf_store import register
from polyrx.config import RootConfig
from polyrx.resources import find_conf_dir


def load_config(overrides: list[str] | None = None) -> RootConfig:
    """Return the composed, type-checked config.

    Parameters
    ----------
    overrides:
        Hydra-style overrides, e.g. ``["meta=ask", "backends.nano.model=gpt-4o"]``.
    """
    register()
    try:
        conf_dir = find_conf_dir()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    with initialize_config_dir(version_base="1.3", config_dir=str(conf_dir)):
        cfg = compose(config_name="config", overrides=overrides or [])
    return OmegaConf.to_object(cfg)  # type: ignore[return-value]

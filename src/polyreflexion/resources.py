"""Locations of packaged data files (prompt templates, config trees).

Everything here goes through :mod:`importlib.resources` rather than
``Path(__file__)`` arithmetic, so the package keeps working when it is
installed as a wheel, zipped, or vendored into another project.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

#: Root of the packaged prompt templates.
PROMPTS_DIR = Path(str(files("polyreflexion") / "prompts"))

#: Root of the packaged Hydra config tree.
CONF_DIR = Path(str(files("polyreflexion") / "conf"))


def prompt_path(name: str) -> Path:
    """Return the path of a packaged prompt file, e.g. ``"reflexion.json"``.

    Raises a clear error rather than failing later inside ``json.load`` when a
    template is missing — a missing file almost always means the wheel was
    built without ``package_data``.
    """
    path = PROMPTS_DIR / name
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in PROMPTS_DIR.glob("*.json")))
        raise FileNotFoundError(f"Packaged prompt {name!r} not found. Available: {available}")
    return path

"""Where the prompt templates and the Hydra config tree live.

These two are located differently on purpose.

*Prompt templates* are packaged. A template is part of what the code does —
changing one changes the method, and a run records each template's sha256 in
its provenance block. They resolve through :mod:`importlib.resources`, so they
keep working from a wheel.

*The config tree* is not packaged. It lives at the repository root as ``conf/``,
because it is the thing a user is expected to read, copy and edit. Hydra reads
it from the working directory rather than from inside the installed package.
"""

from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path

#: Root of the packaged prompt templates.
PROMPTS_DIR = Path(str(files("polyreflexion") / "prompts"))

#: Set this to run against a config tree somewhere other than ``./conf``.
CONF_DIR_ENV = "POLYRX_CONF"

#: Name of the config directory searched for upward from the working directory.
CONF_DIR_NAME = "conf"

#: File that identifies a directory as the config tree rather than some other
#: directory that happens to be called ``conf``.
_CONF_MARKER = "config.yaml"


def prompt_path(name: str) -> Path:
    """Return the path of a packaged prompt file, e.g. ``"reflexion.json"``.

    Raises a clear error rather than failing later inside ``json.load`` when a
    template is missing — that almost always means the wheel was built without
    its package data.
    """
    path = PROMPTS_DIR / name
    if not path.is_file():
        available = ", ".join(sorted(p.name for p in PROMPTS_DIR.glob("*.json")))
        raise FileNotFoundError(f"Packaged prompt {name!r} not found. Available: {available}")
    return path


def find_conf_dir(start: Path | None = None) -> Path:
    """Locate the ``conf/`` tree, searching upward from the working directory.

    Walking up means a run launched from a subdirectory still finds the tree,
    which is the usual case when experimenting inside ``results/`` or a
    notebook folder. ``POLYRX_CONF`` overrides the search outright.
    """
    override = os.environ.get(CONF_DIR_ENV)
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / _CONF_MARKER).is_file():
            raise FileNotFoundError(f"{CONF_DIR_ENV}={override} does not contain {_CONF_MARKER}.")
        return path

    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / CONF_DIR_NAME
        if (candidate / _CONF_MARKER).is_file():
            return candidate

    raise FileNotFoundError(
        f"No {CONF_DIR_NAME}/{_CONF_MARKER} found in {here} or any parent directory.\n"
        f"Run from a checkout of the repository, or point {CONF_DIR_ENV} at a config tree."
    )

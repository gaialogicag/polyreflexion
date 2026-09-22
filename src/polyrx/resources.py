"""Where the Hydra config tree lives.

Nothing in this project is packaged data any more. The config tree — which now
includes every prompt template — sits at the repository root as ``conf/``,
because it is the thing a user is expected to read, copy and edit. It is found
by searching upward from the working directory rather than by looking inside
the installed package.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Set this to run against a config tree somewhere other than ``./conf``.
CONF_DIR_ENV = "POLYRX_CONF"

#: Name of the config directory searched for upward from the working directory.
CONF_DIR_NAME = "conf"

#: File that identifies a directory as the config tree rather than some other
#: directory that happens to be called ``conf``.
_CONF_MARKER = "config.yaml"


def find_conf_dir(start: Path | None = None) -> Path:
    """Locate the ``conf/`` tree, searching upward from the working directory.

    Walking up means a run launched from a subdirectory still finds the tree,
    which is the usual case when working inside ``results/`` or a notebook
    folder. ``POLYRX_CONF`` overrides the search outright.
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

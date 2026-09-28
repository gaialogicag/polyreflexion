"""Where the Hydra config tree lives.

The config tree — which includes every prompt template — is what a user is
expected to read, copy and edit, so a checkout keeps it at the repository root
as ``conf/`` and that copy always wins. A wheel carries a copy of the same tree
as well, because otherwise ``pip install polyreflexion`` produces console
scripts that cannot run: there is no checkout for them to search.

Three tiers, in this order:

1. ``POLYRX_CONF``, which overrides everything.
2. A ``conf/`` directory found by searching upward from the working directory.
   This is the checkout, and it beats the packaged copy so that editing a
   prompt in a checkout changes what a run reads.
3. The copy inside the installed package. ``polyrx-init`` writes it out into
   the working directory for anyone who wants to edit it.
"""

from __future__ import annotations

import importlib.resources
import os
from pathlib import Path

#: Set this to run against a config tree somewhere other than ``./conf``.
CONF_DIR_ENV = "POLYRX_CONF"

#: Name of the config directory searched for upward from the working directory.
CONF_DIR_NAME = "conf"

#: File that identifies a directory as the config tree rather than some other
#: directory that happens to be called ``conf``.
_CONF_MARKER = "config.yaml"


def packaged_conf_dir() -> Path | None:
    """The config tree shipped inside the installed package, if there is one.

    Returns ``None`` for an editable install, where the package directory is
    the source tree and carries no copy — there the checkout's own ``conf/``
    is found by the search instead, which is the right answer anyway.
    """
    try:
        candidate = importlib.resources.files("polyrx") / CONF_DIR_NAME
        marker = candidate / _CONF_MARKER
        if not marker.is_file():
            return None
        # A wheel installed by pip is unpacked onto the filesystem, so this is
        # a real path. Hydra needs one: it cannot read a config tree out of a
        # zip. An import from inside a zip therefore has no packaged tree.
        return Path(str(candidate))
    except (ModuleNotFoundError, TypeError, OSError):
        return None


def install_extra_hint(extra: str) -> str:
    """How to install an optional extra, phrased for the install in use.

    An editable install is a checkout, where the extra has to come from the
    local project; anywhere else it comes from the index. Telling a pip user
    to run ``pip install -e .`` sends them to a directory they do not have.
    """
    if packaged_conf_dir() is None:
        return f'pip install -e ".[{extra}]"'
    return f'pip install "polyreflexion[{extra}]"'


def find_conf_dir(start: Path | None = None) -> Path:
    """Locate the ``conf/`` tree.

    Walking up means a run launched from a subdirectory still finds the tree,
    which is the usual case when working inside ``results/`` or a notebook
    folder. ``POLYRX_CONF`` overrides the search outright, and the copy inside
    the installed package is the last resort.
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

    packaged = packaged_conf_dir()
    if packaged is not None:
        return packaged

    raise FileNotFoundError(
        f"No {CONF_DIR_NAME}/{_CONF_MARKER} found in {here} or any parent directory, "
        f"and the installed package carries no copy.\n"
        f"Run from a checkout of the repository, write an editable tree with "
        f"polyrx-init, or point {CONF_DIR_ENV} at a config tree."
    )

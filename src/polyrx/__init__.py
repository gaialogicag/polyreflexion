"""polyrx — recursive Sierpinski reflexion with a polycontextural meta layer.

The import name is ``polyrx``, matching the ``polyrx-*`` console scripts; the
distribution on disk is ``polyreflexion``.

The public surface is intentionally small: build an engine, optionally wrap it
in a meta controller, and run one of the benchmark suites.  Everything else is
an implementation detail that may change between releases.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

try:  # Installed distribution (the normal case).
    __version__ = _version("polyreflexion")
except PackageNotFoundError:  # Running from a source checkout without install.
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]

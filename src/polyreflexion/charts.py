"""Optional chart rendering.

matplotlib is an optional extra (``pip install polyreflexion[viz]``), because
producing a number does not require drawing it, and a headless benchmark run
should not have to install a plotting stack.

Every chart in this project therefore goes through :func:`pyplot`, which either
returns a headless-configured ``matplotlib.pyplot`` or ``None``. Callers skip
the figure and say so in the report rather than crashing at import time.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

#: Shown once in a report whose charts could not be drawn.
UNAVAILABLE_NOTE = (
    "_Charts omitted: matplotlib is not installed. "
    "Install it with `pip install polyreflexion[viz]` and regenerate the report._"
)


@lru_cache(maxsize=1)
def pyplot() -> Any | None:
    """Return a headless ``matplotlib.pyplot``, or ``None`` when unavailable."""
    try:
        import matplotlib
    except ImportError:
        return None
    # Agg has no display requirement; set before pyplot is first imported.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def available() -> bool:
    """Whether charts can be rendered in this environment."""
    return pyplot() is not None

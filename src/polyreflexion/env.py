"""Load environment variables from a local ``.env`` file.

Only secrets and machine-local endpoints belong in ``.env``.  Everything that
describes *an experiment* belongs in the Hydra config tree instead, so that a
run can be reproduced from the config alone.
"""

from __future__ import annotations

from dotenv import find_dotenv, load_dotenv


def load_env() -> None:
    """Load the nearest ``.env`` found walking up from the working directory.

    ``override=False`` keeps real environment variables authoritative, which is
    what CI and container deployments expect.
    """
    path = find_dotenv(usecwd=True)
    if path:
        load_dotenv(path, override=False)

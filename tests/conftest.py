"""Shared fixtures.

Every test here runs without a network call, an API key or an LLM. The
fixtures below are what make that true: they scrub the environment variables
the package reads, and they restore the module-level client registry that the
entry points would otherwise leave installed for the next test.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from polyrx.models.base import CallableLLMClient
from polyrx.resources import CONF_DIR_ENV

#: Environment variables that change what the package resolves. A developer
#: machine has several of these set for real, and a test that reads one of
#: them would pass locally and fail in continuous integration, or the reverse.
_SCRUBBED = (CONF_DIR_ENV, "OPENAI_API_KEY", "GEMINI_API_KEY")


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the environment variables the package reads.

    Autouse, because the failure it prevents is silent: a test that forgot to
    scrub ``OPENAI_API_KEY`` passes on a machine that has one and fails
    nowhere else.
    """
    for name in _SCRUBBED:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def conf_tree(tmp_path: Path) -> Callable[[str], Path]:
    """Build a directory that ``find_conf_dir`` will accept as a config tree.

    Only the marker file has to exist for the search to succeed, so the
    factory writes that and nothing else.
    """

    def _make(relative: str = "conf") -> Path:
        directory = tmp_path / relative
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "config.yaml").write_text("# test config tree\n")
        return directory

    return _make


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A working directory with no config tree in it or in any parent.

    ``tmp_path`` is under the system temporary directory, never under a
    checkout, which is exactly the situation of a user who installed the
    package from PyPI.
    """
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


@pytest.fixture
def stub_client() -> CallableLLMClient:
    """An LLM client that answers deterministically and never leaves the process."""
    return CallableLLMClient(lambda prompt: f"stub response to: {prompt[:40]}")


@pytest.fixture
def restore_active_registry() -> Iterator[None]:
    """Undo ``set_active_backends``.

    The registry is a module-level global. A test that installs one would
    otherwise decide which clients every later test builds.
    """
    from polyrx.models import registry

    saved = registry._active
    yield
    registry._active = saved

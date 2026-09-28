"""Config tree resolution.

``find_conf_dir`` decides which ``conf/`` tree a run reads, and every entry
point goes through it. Three tiers: the ``POLYRX_CONF`` override, a search
upward from the working directory, then the copy inside the installed package.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from polyrx import resources
from polyrx.resources import (
    CONF_DIR_ENV,
    find_conf_dir,
    install_extra_hint,
    packaged_conf_dir,
)


class TestEnvironmentOverride:
    def test_override_is_used(
        self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        elsewhere = conf_tree("somewhere/else")
        monkeypatch.setenv(CONF_DIR_ENV, str(elsewhere))

        assert find_conf_dir() == elsewhere.resolve()

    def test_override_beats_a_tree_in_the_working_directory(
        self,
        conf_tree: Callable[[str], Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The override is absolute: a checkout underfoot does not win."""
        local = conf_tree("checkout/conf")
        override = conf_tree("override/conf")
        monkeypatch.chdir(local.parent)
        monkeypatch.setenv(CONF_DIR_ENV, str(override))

        assert find_conf_dir() == override.resolve()

    def test_override_expands_a_home_relative_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        home = tmp_path / "home"
        (home / "myconf").mkdir(parents=True)
        (home / "myconf" / "config.yaml").write_text("# tree\n")
        monkeypatch.setenv("HOME", str(home))
        monkeypatch.setenv(CONF_DIR_ENV, "~/myconf")

        assert find_conf_dir() == (home / "myconf").resolve()

    def test_override_without_the_marker_is_an_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pointing the override at the wrong directory must say so, not fall
        back to a search that would quietly read a different tree."""
        empty = tmp_path / "not-a-conf-tree"
        empty.mkdir()
        monkeypatch.setenv(CONF_DIR_ENV, str(empty))

        with pytest.raises(FileNotFoundError) as excinfo:
            find_conf_dir()

        message = str(excinfo.value)
        assert CONF_DIR_ENV in message
        assert "config.yaml" in message


class TestUpwardSearch:
    def test_tree_in_the_working_directory(
        self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = conf_tree("checkout/conf")
        monkeypatch.chdir(tree.parent)

        assert find_conf_dir() == tree.resolve()

    def test_tree_found_from_a_nested_subdirectory(
        self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run launched from ``results/`` still finds the checkout's tree."""
        tree = conf_tree("checkout/conf")
        nested = tree.parent / "results" / "run-01" / "cache"
        nested.mkdir(parents=True)
        monkeypatch.chdir(nested)

        assert find_conf_dir() == tree.resolve()

    def test_start_argument_is_searched_instead_of_the_working_directory(
        self, conf_tree: Callable[[str], Path], isolated_cwd: Path
    ) -> None:
        tree = conf_tree("checkout/conf")

        assert find_conf_dir(start=tree.parent) == tree.resolve()

    def test_a_conf_directory_without_the_marker_is_skipped(
        self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``conf/`` is a common directory name. Only the one carrying
        ``config.yaml`` is this project's tree."""
        real = conf_tree("checkout/conf")
        decoy = real.parent / "service" / "conf"
        decoy.mkdir(parents=True)
        monkeypatch.chdir(decoy.parent)

        assert find_conf_dir() == real.resolve()


class TestNothingFound:
    def test_error_names_the_directory_searched(self, isolated_cwd: Path) -> None:
        with pytest.raises(FileNotFoundError) as excinfo:
            find_conf_dir()

        message = str(excinfo.value)
        assert str(isolated_cwd.resolve()) in message
        assert CONF_DIR_ENV in message

    def test_error_mentions_both_remedies(self, isolated_cwd: Path) -> None:
        with pytest.raises(FileNotFoundError, match="polyrx-init"):
            find_conf_dir()


class TestPackagedFallback:
    """The tier that makes ``pip install polyreflexion`` usable.

    An editable install has no packaged copy — the package directory is the
    source tree — so these tests install one rather than depending on how the
    environment running them was built. The wheel itself is exercised by the
    continuous integration job that installs it into an empty directory.
    """

    @pytest.fixture
    def packaged(self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch) -> Path:
        tree = conf_tree("site-packages/polyrx/conf")
        monkeypatch.setattr(resources, "packaged_conf_dir", lambda: tree)
        return tree

    def test_used_when_no_checkout_is_found(self, packaged: Path, isolated_cwd: Path) -> None:
        assert find_conf_dir() == packaged

    def test_a_checkout_wins_over_the_packaged_copy(
        self, packaged: Path, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Editing a prompt in a checkout has to change what a run reads. If
        the packaged copy won, the edit would be silently ignored."""
        checkout = conf_tree("checkout/conf")
        monkeypatch.chdir(checkout.parent)

        assert find_conf_dir() == checkout.resolve()

    def test_the_override_wins_over_the_packaged_copy(
        self,
        packaged: Path,
        conf_tree: Callable[[str], Path],
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        override = conf_tree("override/conf")
        monkeypatch.setenv(CONF_DIR_ENV, str(override))

        assert find_conf_dir() == override.resolve()


class TestPackagedConfDir:
    def test_absent_without_a_packaged_copy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An editable install points at the source tree, which carries no
        copy. Returning None there is what sends the search to the checkout."""
        monkeypatch.setattr(resources.importlib.resources, "files", lambda _: Path("/nonexistent"))

        assert packaged_conf_dir() is None

    def test_found_when_the_package_carries_the_marker(
        self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tree = conf_tree("site-packages/polyrx/conf")
        monkeypatch.setattr(resources.importlib.resources, "files", lambda _: tree.parent)

        assert packaged_conf_dir() == tree

    def test_an_unreadable_package_location_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A tree that cannot be read as files -- an import from inside a zip
        -- has no usable config tree, and Hydra could not read it anyway."""

        def explode(_: str) -> object:
            raise OSError("not a filesystem path")

        monkeypatch.setattr(resources.importlib.resources, "files", explode)

        assert packaged_conf_dir() is None


class TestInstallExtraHint:
    """Telling a pip user to run ``pip install -e .`` sends them to a directory
    they do not have, and telling a developer to install from the index would
    replace their editable checkout."""

    def test_an_editable_install_is_told_to_use_the_checkout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resources, "packaged_conf_dir", lambda: None)

        assert install_extra_hint("viz") == 'pip install -e ".[viz]"'

    def test_an_installed_package_is_told_to_use_the_index(
        self, conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(resources, "packaged_conf_dir", lambda: conf_tree("pkg/conf"))

        assert install_extra_hint("viz") == 'pip install "polyreflexion[viz]"'

    def test_the_extra_is_named(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(resources, "packaged_conf_dir", lambda: None)

        assert "gemini" in install_extra_hint("gemini")

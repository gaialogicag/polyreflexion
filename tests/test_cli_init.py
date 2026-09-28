"""``polyrx-init``: write the packaged config tree out where it can be edited."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from polyrx import resources
from polyrx.cli.init import main
from polyrx.resources import find_conf_dir


@pytest.fixture
def packaged(conf_tree: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch) -> Path:
    """A packaged tree with a nested file, so copying is exercised properly."""
    tree = conf_tree("site-packages/polyrx/conf")
    (tree / "prompts" / "meta").mkdir(parents=True)
    (tree / "prompts" / "meta" / "default.yaml").write_text("drift_check: original\n")
    monkeypatch.setattr(resources, "packaged_conf_dir", lambda: tree)
    return tree


class TestWriting:
    def test_writes_the_tree_into_the_working_directory(
        self, packaged: Path, isolated_cwd: Path
    ) -> None:
        assert main([]) == 0
        assert (isolated_cwd / "conf" / "config.yaml").is_file()

    def test_copies_nested_files(self, packaged: Path, isolated_cwd: Path) -> None:
        main([])

        written = isolated_cwd / "conf" / "prompts" / "meta" / "default.yaml"
        assert written.read_text() == "drift_check: original\n"

    def test_dest_overrides_the_working_directory(
        self, packaged: Path, isolated_cwd: Path, tmp_path: Path
    ) -> None:
        elsewhere = tmp_path / "chosen"

        assert main(["--dest", str(elsewhere)]) == 0
        assert (elsewhere / "config.yaml").is_file()

    def test_the_written_tree_is_what_a_run_then_reads(
        self, packaged: Path, isolated_cwd: Path
    ) -> None:
        """The point of the command. A tree nobody's run reads is useless."""
        main([])

        assert find_conf_dir() == (isolated_cwd / "conf").resolve()

    def test_reports_how_many_files_it_wrote(
        self, packaged: Path, isolated_cwd: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main([])

        assert "2 files" in capsys.readouterr().out


class TestRefusingToClobber:
    def test_an_existing_tree_is_left_alone(self, packaged: Path, isolated_cwd: Path) -> None:
        """Edits are the reason the tree was written out. Overwriting them by
        default would destroy the only copy."""
        existing = isolated_cwd / "conf"
        existing.mkdir()
        (existing / "config.yaml").write_text("edited by hand\n")

        assert main([]) == 1
        assert (existing / "config.yaml").read_text() == "edited by hand\n"

    def test_force_overwrites(self, packaged: Path, isolated_cwd: Path) -> None:
        existing = isolated_cwd / "conf"
        existing.mkdir()
        (existing / "config.yaml").write_text("edited by hand\n")

        assert main(["--force"]) == 0
        assert (existing / "config.yaml").read_text() == "# test config tree\n"

    def test_force_removes_files_the_packaged_tree_does_not_have(
        self, packaged: Path, isolated_cwd: Path
    ) -> None:
        existing = isolated_cwd / "conf"
        existing.mkdir()
        (existing / "stale.yaml").write_text("left over\n")

        main(["--force"])

        assert not (existing / "stale.yaml").exists()


class TestNothingToWrite:
    def test_an_editable_install_says_so_instead_of_failing_obscurely(
        self,
        isolated_cwd: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(resources, "packaged_conf_dir", lambda: None)

        assert main([]) == 1
        assert "editable install" in capsys.readouterr().err

    def test_refuses_to_copy_the_packaged_tree_onto_itself(
        self, packaged: Path, isolated_cwd: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--dest", str(packaged)]) == 1
        assert "packaged tree itself" in capsys.readouterr().err

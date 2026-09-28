"""Write the packaged config tree into the working directory.

A wheel carries a copy of ``conf/`` so that an installed console script can
run at all, but a copy inside site-packages is not somewhere anyone edits. This
writes it out where it can be read, changed and kept under version control.
The written tree then wins over the packaged one, because
:func:`polyrx.resources.find_conf_dir` searches upward from the working
directory before falling back to the package.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from polyrx import resources
from polyrx.resources import CONF_DIR_NAME


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyrx-init",
        description="Write the packaged config tree into the working directory so it can be edited.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=None,
        help=f"Where to write the tree. Default: ./{CONF_DIR_NAME}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing tree. Without this, an existing directory is left alone.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dest = (args.dest or Path.cwd() / CONF_DIR_NAME).resolve()

    # Through the module, not a bound name: this and find_conf_dir then
    # resolve the packaged tree the same way.
    source = resources.packaged_conf_dir()
    if source is None:
        print(
            "This install carries no packaged config tree, which is normal for an "
            "editable install from a checkout: the repository's own conf/ is already "
            "the tree every run reads.",
            file=sys.stderr,
        )
        return 1

    if source == dest:
        print(f"{dest} is the packaged tree itself. Choose a different --dest.", file=sys.stderr)
        return 1

    if dest.exists():
        if not args.force:
            # Refusing is the whole point: the tree is edited, and the edits
            # are the reason someone wrote it out in the first place.
            print(
                f"{dest} already exists. Leave it, or pass --force to overwrite it.",
                file=sys.stderr,
            )
            return 1
        shutil.rmtree(dest)

    shutil.copytree(source, dest)
    count = sum(1 for path in dest.rglob("*") if path.is_file())
    print(f"Wrote {count} files to {dest}")
    print("Runs launched from here, or from any directory below it, now read this tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

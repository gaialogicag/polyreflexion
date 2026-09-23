"""Inspect the experiment grid and manage its lock file.

The grid lives in ``conf/conditions/``. ``data/conditions.lock.json`` records
the grid the published results were produced with, in the same spirit as
``data/MANIFEST.json`` for datasets: every cached summary is keyed by a
condition's ``summary_key``, so changing one orphans the cache silently rather
than failing. Runs compare the two and stop when they disagree.
"""

from __future__ import annotations

import argparse
import sys

from polyrx.cli.compose import load_config
from polyrx.conditions import (
    LOCK_FILENAME,
    check_against_lock,
    registry_from_config,
    write_lock,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyrx-conditions",
        description="Show the experiment grid, or lock it against accidental change.",
    )
    parser.add_argument(
        "action",
        choices=("show", "lock", "check"),
        help="show: print the grid; lock: record it; check: compare against the lock",
    )
    parser.add_argument(
        "overrides", nargs="*", help="Hydra overrides, e.g. conditions=answerer_only"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = load_config(args.overrides)
    registry = registry_from_config(cfg.conditions.conditions)
    lock_path = cfg.paths.resolved("data_dir") / LOCK_FILENAME

    if args.action == "show":
        header = f"{'condition':22s} {'backend':8s} {'depth':>5s} {'cycles':>6s}  {'cache key':16s} continues from"
        print(header)
        print("-" * len(header))
        for name in registry.names():
            c = registry.get(name)
            print(
                f"{c.name:22s} {c.backend:8s} {c.depth:5d} {c.meta_cycles:6d}  "
                f"{c.summary_key or '-':16s} {c.prior_summary_key or '-'}"
            )
        return 0

    if args.action == "lock":
        write_lock(registry, lock_path)
        print(f"Locked {len(registry.names())} conditions to {lock_path}")
        return 0

    problems = check_against_lock(registry, lock_path)
    if not lock_path.is_file():
        print(f"No lock at {lock_path}. Create one with: polyrx-conditions lock")
        return 1
    if problems:
        print(f"The grid no longer matches {lock_path}:")
        for problem in problems:
            print(f"  - {problem}")
        print(
            "\nEvery cached summary is keyed by summary_key, so a mismatch orphans the "
            "cache. Re-lock deliberately with: polyrx-conditions lock"
        )
        return 1
    print(f"Grid matches {lock_path} ({len(registry.names())} conditions).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

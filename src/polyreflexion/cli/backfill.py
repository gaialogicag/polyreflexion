"""Backfill polycontextural judge traces for an existing OpenToM benchmark run.

Summaries cached before the judge traces were recorded have no ``.meta.json``
beside them, which is what the dimension-attribution section of the detailed
report reads. This re-runs only the meta evaluation for those summaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyreflexion.benchmark.opentom_loader import load_items
from polyreflexion.benchmark.runner import backfill_meta_traces, load_run
from polyreflexion.cli.compose import load_config
from polyreflexion.engine import PromptRegistry
from polyreflexion.models.registry import set_active_backends


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write .meta.json judge traces beside cached meta summaries"
    )
    parser.add_argument("run_json", type=Path, help="Path to opentom_*.json results file")
    parser.add_argument(
        "--summary-key",
        default="",
        help="Only backfill one summary key (e.g. nano_meta_c2)",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing traces")
    parser.add_argument("--stub", action="store_true", help="Use stub meta clients")
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. backends.meta_judge.model=gpt-4o-mini",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.overrides)
    set_active_backends(cfg.backends)
    if not args.stub:
        # Fails with the variable name when the key is missing.
        cfg.backends.meta_judge.api_key()

    run = load_run(args.run_json)
    story_ids = sorted(run.summaries.keys())
    narratives = {item.story_id: item.narrative for item in load_items() if item.story_id in story_ids}
    missing = [sid for sid in story_ids if sid not in narratives]
    if missing:
        raise SystemExit(f"Could not load narratives for stories: {missing[:5]}")

    keys = (args.summary_key,) if args.summary_key else None
    config = run.config
    if args.stub:
        config.use_stub = True

    written = backfill_meta_traces(
        narratives,
        config,
        PromptRegistry(),
        summary_keys=keys,
        force=args.force,
    )
    print(f"Wrote {written} meta trace file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

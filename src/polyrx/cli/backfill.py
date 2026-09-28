"""Backfill polycontextural judge traces for an existing benchmark run.

Summaries cached before the judge traces were recorded have no ``.meta.json``
beside them, which is what the dimension-attribution section of the detailed
report reads. This re-runs only the meta evaluation for those summaries.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyrx.benchmark.runner import backfill_meta_traces, load_dataset_items, load_run
from polyrx.cli.compose import load_config
from polyrx.conditions import registry_from_config
from polyrx.config import HostedModelConfig
from polyrx.engine import PromptRegistry
from polyrx.models.registry import set_active_backends


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write .meta.json judge traces beside cached meta summaries"
    )
    parser.add_argument("run_json", type=Path, help="Path to a saved results JSON file")
    parser.add_argument(
        "--summary-key",
        default="",
        help="Only backfill one summary key (e.g. answerer_meta_c2)",
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
    set_active_backends(cfg.backends, cfg.postprocess)
    if not args.stub and isinstance(cfg.backends.meta_judge, HostedModelConfig):
        # Fails with the variable name when the key is missing. A meta judge on
        # a local runtime has no key to check.
        cfg.backends.meta_judge.api_key()

    run = load_run(args.run_json)
    config = run.config
    if args.stub:
        config.use_stub = True
    # A saved run records the prompt set's name, not its text, so the templates
    # come from the current config tree. Pass overrides to reproduce an older
    # set exactly, e.g. `prompts/meta=exploretom`.
    config.prompts = cfg.prompts
    # A saved run records its condition names, not the grid that defines them,
    # so the grid is rebuilt from the current config tree. Without it every
    # lookup of a condition's backend, depth and cache key fails.
    config.condition_registry = registry_from_config(cfg.conditions.conditions)

    # The contexts come from the run's own dataset, not from whatever dataset
    # the config tree currently defaults to: a trace written against a
    # different story than the summary was built from describes nothing.
    # Asking for exactly the ids the run summarized makes the sample explicit,
    # so a dataset that has since gained or lost items cannot silently
    # substitute a different draw.
    item_ids = sorted(run.summaries)
    config.only_item_ids = frozenset(item_ids)
    items, _ = load_dataset_items(config)
    contexts = {item.item_id: item.context for item in items if item.item_id in run.summaries}

    missing = [item_id for item_id in item_ids if item_id not in contexts]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(item_ids)} items in {args.run_json.name} are not in "
            f"dataset {config.dataset.name!r}: {missing[:5]}. The run was scored against a "
            f"different dataset, or against a revision this one no longer pins."
        )

    keys = (args.summary_key,) if args.summary_key else None
    written = backfill_meta_traces(
        contexts,
        config,
        PromptRegistry(cfg.prompts.reflexion),
        summary_keys=keys,
        force=args.force,
    )
    print(f"Wrote {written} meta trace file(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

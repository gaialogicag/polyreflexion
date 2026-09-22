"""CLI entry point for the polycontextural meta-evaluation layer.

Runs the recursive reflexion engine under the meta controller: each cycle is
judged from three logical contexts (subjective / objective / dialectical), the
triple steers expansion / refinement / reset, and the run ends with a selected
answer plus a full trace and Sierpinski topology renderings.

Examples:

    polyrx-reflect "Why do people keep secrets?"
    polyrx-reflect --stub "smoke test without any API"
    polyrx-reflect --max-cycles 4 --depth 2 "Question..."

Anything after the flags is a Hydra override, so the backends come from the
same config tree the benchmarks use::

    polyrx-reflect "Question..." backends.nano.model=gpt-4o-mini meta=gpqa
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from polyrx.cli.compose import load_config
from polyrx.engine import PromptRegistry, ReflexionEngine, StubLLMClient
from polyrx.meta.datatypes import MetaConfig, MetaResult
from polyrx.meta.factory import build_meta_controller
from polyrx.meta.judges import StubMetaClient
from polyrx.meta.prompts import MetaPromptRegistry
from polyrx.meta.trace import MetaTrace
from polyrx.meta.viz import render_topology
from polyrx.models.base import LLMClient
from polyrx.models.registry import get_client, set_active_backends


def print_overview(result: MetaResult) -> None:
    """Console table of the run: one row per cycle plus the final answer."""
    print("\n" + "=" * 72)
    print("META-REFLEXION OVERVIEW")
    print("=" * 72)
    print(f"Question:    {result.question[:200]}")
    print(f"Termination: {result.termination_reason}  |  selected cycle: {result.selected_cycle}")
    print("-" * 72)
    print(f"{'cycle':>5}  {'triple':<9} {'global':<6} {'action':<10} {'drift':<5} path")
    for cycle in result.cycles:
        path = " > ".join(cycle.topology.path) or "root"
        print(
            f"{cycle.index:>5}  {cycle.evaluation.as_triple():<9} "
            f"{cycle.global_value.value:<6} {cycle.action or '-':<10} "
            f"{'yes' if cycle.drifted else 'no':<5} {path}"
        )
    print("-" * 72)
    print("FINAL ANSWER:\n")
    print(result.final_answer or "(no answer)")
    print("=" * 72)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polyrx-reflect", description="Run polycontextural meta-reflexion"
    )
    parser.add_argument("text", nargs="?", help="Question or input text (prompted if omitted)")
    parser.add_argument("--max-cycles", type=int, default=None, help="Meta cycle budget")
    parser.add_argument("--depth", type=int, default=None, help="Reflexion depth per cycle")
    parser.add_argument("--engine", default="nano", help="Engine backend role (default nano)")
    parser.add_argument(
        "--judge", default="meta_judge", help="Judge backend role (default meta_judge)"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--stub", action="store_true", help="Deterministic stub clients (no API)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides (key=value)")
    args = parser.parse_args(argv)

    text = args.text or input("Question: ").strip()
    if not text:
        raise SystemExit("No input text given.")

    cfg = load_config(args.overrides)
    set_active_backends(cfg.backends)
    # Command-line flags win over the config, which wins over the defaults.
    max_cycles = args.max_cycles if args.max_cycles is not None else cfg.meta.max_cycles
    depth = args.depth if args.depth is not None else cfg.meta.engine_depth
    workers = args.workers if args.workers is not None else cfg.engine.max_workers
    output = args.output or (cfg.paths.resolved("results_dir") / "meta")

    engine_client: LLMClient
    judge_client: LLMClient
    if args.stub:
        engine_client = StubLLMClient()
        # Scripted path: refine on cycle 0, expand on cycle 1, converge after.
        judge_client = StubMetaClient(triples=[("A", "F", "R"), ("A", "W", "R"), ("A", "W", "W")])
    else:
        engine_client = get_client(args.engine)
        judge_client = get_client(args.judge)

    prompts = PromptRegistry(cfg.prompts.reflexion)

    def engine_factory(depth: int) -> ReflexionEngine:
        return ReflexionEngine(engine_client, prompts=prompts, max_depth=depth, max_workers=workers)

    controller = build_meta_controller(
        engine_factory,
        judge_client,
        config=MetaConfig(max_cycles=max_cycles, engine_depth=depth),
        meta_prompts=MetaPromptRegistry(cfg.prompts.meta),
    )

    print(
        f"Running meta-reflexion (engine={args.engine}, judges={args.judge}, "
        f"max_cycles={max_cycles}, depth={depth})..."
    )
    result = controller.run(text)
    print_overview(result)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output / stamp
    render_topology(result, run_dir)  # render first so the report embeds the PNGs
    report_path = MetaTrace(run_dir).save(result)
    print(f"\nSaved run artifacts to {run_dir}/")
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Ask one question and compare plain reflexion against meta-guided reflexion.

Pipeline (per the project spec):

1. **Baseline reflexion** runs on the nano OpenAI model (`OPENAI_MODEL_NANO`,
   default gpt-5.4-nano) — this is cycle 0 of the meta run.
2. The three **polycontextural judges** run on the meta judge model
   (`OPENAI_MODEL_META_JUDGE`, default gpt-4o — deliberately *not* the OpenToM
   benchmark judge).
3. Every follow-up **meta cycle** (expansion / refinement / reset) runs on nano
   again.

Prompts for this script are answer-oriented (``meta/ask_prompts.py``): the
engine uses the three perspectives as internal reasoning, then emits the
*single best answer* to the question — not a summary of competing views.
OpenToM story-summary templates are left unchanged.

Usage:

    polyrx-ask "Why do people keep secrets?"
    polyrx-ask                    # prompts interactively
    polyrx-ask --stub "offline smoke test"
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from datetime import UTC, datetime
from pathlib import Path

from polyreflexion.cli.compose import load_config
from polyreflexion.engine import ReflexionEngine, StubLLMClient
from polyreflexion.meta.ask_prompts import build_ask_engine_prompts, build_ask_meta_prompts
from polyreflexion.meta.datatypes import MetaConfig, MetaResult
from polyreflexion.meta.factory import build_meta_controller
from polyreflexion.meta.judges import StubMetaClient
from polyreflexion.meta.trace import MetaTrace
from polyreflexion.meta.viz import render_topology
from polyreflexion.models.registry import get_client, set_active_backends


def _wrap(text: str, width: int = 76) -> str:
    """Wrap long answers for readable console output, keeping paragraphs."""
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    return "\n".join(textwrap.fill(p, width=width) for p in paragraphs) or "(empty)"


def print_comparison(result: MetaResult) -> None:
    """Side-by-side style overview: baseline reflexion vs. meta reflexion."""
    baseline = result.cycles[0] if result.cycles else None
    selected = next((c for c in result.cycles if c.index == result.selected_cycle), None)

    print("\n" + "=" * 78)
    print("QUESTION")
    print("=" * 78)
    print(_wrap(result.question))

    print("\n" + "=" * 78)
    print("1) BASELINE REFLEXION (nano, cycle 0)")
    print("=" * 78)
    if baseline is None:
        print("(engine failed before producing a baseline answer)")
    else:
        print(_wrap(baseline.result.summary))
        print(
            f"\nJudges (gpt-4o): {baseline.evaluation.as_triple()} -> "
            f"global {baseline.global_value.value}, action: {baseline.action or '-'}"
        )

    print("\n" + "=" * 78)
    print("2) META REFLEXION (nano cycles steered by the meta layer)")
    print("=" * 78)
    print(_wrap(result.final_answer))
    if selected is not None:
        print(
            f"\nSelected cycle {selected.index} of {len(result.cycles)} "
            f"({selected.evaluation.as_triple()}, "
            f"{selected.positives_count()} positive judgments); "
            f"termination: {result.termination_reason}"
        )

    print("\n" + "-" * 78)
    print("CYCLE HISTORY")
    print("-" * 78)
    print(f"{'cycle':>5}  {'triple':<9} {'global':<6} {'action':<10} {'drift':<5} path")
    for cycle in result.cycles:
        marker = " *" if cycle.index == result.selected_cycle else ""
        path = " > ".join(cycle.topology.path) or "root"
        print(
            f"{cycle.index:>5}  {cycle.evaluation.as_triple():<9} "
            f"{cycle.global_value.value:<6} {cycle.action or '-':<10} "
            f"{'yes' if cycle.drifted else 'no':<5} {path}{marker}"
        )
    if result.cycles:
        same = result.selected_cycle == 0
        print(
            "\nComparison: the meta layer "
            + (
                "kept the baseline answer."
                if same
                else f"replaced the baseline with cycle {result.selected_cycle}."
            )
        )
    print("-" * 78)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polyrx-ask",
        description="Compare baseline reflexion against meta reflexion for one question",
    )
    parser.add_argument("question", nargs="?", help="Question (prompted if omitted)")
    parser.add_argument("--max-cycles", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None, help="Reflexion depth per cycle")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stub", action="store_true", help="Offline stub clients")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides (key=value)")
    args = parser.parse_args(argv)

    question = args.question or input("Question: ").strip()
    if not question:
        raise SystemExit("No question given.")

    # The ask profile uses answer-oriented templates; override it if you want
    # the benchmark-style summary prompts instead.
    cfg = load_config(["meta=ask", *args.overrides])
    set_active_backends(cfg.backends)
    max_cycles = args.max_cycles if args.max_cycles is not None else cfg.meta.max_cycles
    workers = args.workers if args.workers is not None else cfg.engine.max_workers
    depth = args.depth if args.depth is not None else cfg.meta.engine_depth
    output = args.output or (cfg.paths.resolved("results_dir") / "meta")

    if args.stub:
        engine_client = StubLLMClient()
        judge_client = StubMetaClient(triples=[("A", "F", "R"), ("A", "W", "W")])
    else:
        engine_client = get_client("nano")  # reflexion + meta cycles
        judge_client = get_client("meta_judge")  # polycontextural judges

    # Answer-oriented templates: best reply to the question, not a survey of views.
    # (OpenToM keeps the story-summary prompts in prompts.json.)
    engine_prompts = build_ask_engine_prompts()
    meta_prompts = build_ask_meta_prompts()

    def engine_factory(depth: int) -> ReflexionEngine:
        return ReflexionEngine(
            engine_client,
            prompts=engine_prompts,
            max_depth=depth,
            max_workers=workers,
        )

    controller = build_meta_controller(
        engine_factory,
        judge_client,
        config=MetaConfig(max_cycles=max_cycles, engine_depth=depth),
        meta_prompts=meta_prompts,
    )

    print(
        "Running baseline reflexion + meta cycles "
        f"(engine={cfg.backends.nano.model}, judges={cfg.backends.meta_judge.model}, "
        f"max_cycles={max_cycles})..."
    )
    result = controller.run(question)
    print_comparison(result)

    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = output / f"ask_{stamp}"
    render_topology(result, run_dir)
    report_path = MetaTrace(run_dir).save(result)
    print(f"\nSaved full trace and report to {run_dir}/ ({report_path.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

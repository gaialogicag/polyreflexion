"""Does the geometric surplus -- the Refine cycle's unused apex -- beat the
canonical one, if actually pursued?

For every cycle where the meta-controller judges exactly one dimension
positive (``GlobalValue.A``, a Refine step), the controller descends into
that positive dimension's corner (the "canonical" arrangement). The geometry
map also records, purely as bookkeeping, an unused "surplus" arrangement:
descending into one of the two *negative*-judged dimensions instead
(``polyrx/meta/geometry.py``). Nothing in the pipeline ever executes that
branch -- ``MetaController``'s ``StrategyPlanner`` only ever receives
``(global_value, evaluation)``, never the topology/surplus record
(``controller.py``). The surplus is never fed into a prompt, never judged,
never compared to anything.

This script builds that branch for real. For every Refine cycle it
constructs the exact ``refine_input`` prompt the real pipeline would build,
but with the surplus-implied dimension as the apex instead of the real one
(reusing ``PolyEvaluation.guidance_for()``, which already handles negative
dimensions honestly: it returns "F (negative): <rationale>" rather than
pretending a rejected dimension is a strength). The two other dimensions'
existing rationales become the boundaries, exactly as ``reuse_negatives()``
does for the real apex. One new engine + judge call executes that branch; the
real (canonical) continuation is untouched and keeps running normally.

Compared: the counterfactual branch's judged-positivity count against what
the real path actually achieved on its next cycle. Scope is deliberately
limited to Refine (A) cycles -- Expand (R) and the constant W/F surpluses
don't map cleanly onto any existing prompt mechanism (see the ablation
writeup for why).

Run with the project's own config tree, e.g.

    python surplus_counterfactual.py --dataset exploretom --stories 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from polyrx.benchmark.runner import BenchmarkConfig, load_dataset_items
from polyrx.cli.compose import load_config
from polyrx.engine import PromptRegistry, ReflexionEngine
from polyrx.meta.datatypes import (
    DIMENSION_ORDER,
    DIMENSION_TO_VERTEX,
    Dimension,
    GlobalValue,
    MetaConfig,
    Observation,
)
from polyrx.meta.factory import build_meta_controller
from polyrx.meta.judges import PolyJudge
from polyrx.meta.prompts import MetaPromptRegistry
from polyrx.models.registry import get_client, set_active_backends

VERTEX_TO_DIMENSION = {v: k for k, v in DIMENSION_TO_VERTEX.items()}


def surplus_dimension_for(cycle) -> Dimension:
    """The negative-judged dimension the geometry map recorded as unused."""
    assert cycle.topology.surplus.kind == "descend_corner"
    (vertex,) = cycle.topology.surplus.anchor
    return VERTEX_TO_DIMENSION[vertex]


def build_counterfactual_refine_input(
    meta_prompts: MetaPromptRegistry,
    *,
    question: str,
    answer: str,
    evaluation,
    surplus_dim: Dimension,
) -> str:
    """The same refine_input template the real pipeline uses, apex swapped."""
    by_dim = {j.dimension: j for j in evaluation.judgments()}
    others = [d for d in DIMENSION_ORDER if d != surplus_dim]
    left_dim, right_dim = others[0], others[1]
    return meta_prompts.refine_input(
        question=question,
        answer=answer,
        apex=DIMENSION_TO_VERTEX[surplus_dim].value,
        apex_guidance=evaluation.guidance_for(surplus_dim),
        left_label=left_dim.value,
        left_boundary=by_dim[left_dim].rationale,
        right_label=right_dim.value,
        right_boundary=by_dim[right_dim].rationale,
        judge_feedback=evaluation.feedback_block(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="exploretom")
    ap.add_argument("--stories", type=int, default=5)
    ap.add_argument("--max-cycles", type=int, default=4)
    ap.add_argument("--engine-depth", type=int, default=1)
    ap.add_argument("--out", default="surplus_counterfactual.json")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    overrides = [f"experiment={args.dataset}", "backends=openai_nano", *args.overrides]
    cfg = load_config(overrides)
    set_active_backends(cfg.backends, cfg.postprocess)

    prompts = PromptRegistry(cfg.prompts.reflexion)
    meta_prompts = MetaPromptRegistry(cfg.prompts.meta)
    client = get_client("answerer")
    judge_client = get_client("meta_judge")
    judge = PolyJudge(judge_client, prompts=meta_prompts)

    def engine_factory(depth: int) -> ReflexionEngine:
        return ReflexionEngine(
            client, prompts=prompts, max_depth=depth, max_workers=cfg.experiment.max_workers
        )

    bench_cfg = BenchmarkConfig(
        dataset=cfg.dataset,
        num_items=args.stories,
        seed=cfg.experiment.seed,
        max_workers=cfg.experiment.max_workers,
        conditions=("answerer_direct",),
        use_stub=False,
        cache_namespace="",
        prompts=cfg.prompts,
        paths=cfg.paths,
        report=cfg.report,
        postprocess=cfg.postprocess,
    )
    items, _ = load_dataset_items(bench_cfg)

    seen: dict[str, str] = {}
    for item in items:
        if item.context not in seen.values():
            seen[item.item_id] = item.context
        if len(seen) >= args.stories:
            break
    stories = list(seen.items())
    print(f"{len(stories)} distinct stories from {args.dataset}")

    rows: list[dict] = []
    for n, (sid, text) in enumerate(stories, 1):
        controller = build_meta_controller(
            engine_factory,
            judge_client,
            config=MetaConfig(max_cycles=args.max_cycles, engine_depth=args.engine_depth),
            meta_prompts=meta_prompts,
        )
        result = controller.run(text)
        cycles = result.cycles
        a_cycles = [i for i, c in enumerate(cycles) if c.global_value is GlobalValue.A]
        print(f"  [{n}/{len(stories)}] {sid}: {len(cycles)} cycles, {len(a_cycles)} Refine cycles")

        for i in a_cycles:
            cycle = cycles[i]
            evaluation = cycle.evaluation
            (positive_dim,) = evaluation.positives()
            surplus_dim = surplus_dimension_for(cycle)

            counterfactual_input = build_counterfactual_refine_input(
                meta_prompts,
                question=text,
                answer=cycle.result.summary,
                evaluation=evaluation,
                surplus_dim=surplus_dim,
            )
            with engine_factory(args.engine_depth) as engine:
                counterfactual_result = engine.run(counterfactual_input)
            counterfactual_eval = judge.evaluate(
                Observation(
                    input_text=text,
                    summary=counterfactual_result.summary,
                    corner_answers=counterfactual_result.corner_answers,
                    previous_summary=cycle.result.summary,
                )
            )

            has_next = i + 1 < len(cycles)
            row = {
                "story_id": sid,
                "cycle_index": i,
                "positive_dim": positive_dim.value,
                "surplus_dim": surplus_dim.value,
                "canonical_positives_this_cycle": len(evaluation.positives()),
                "counterfactual_positives": len(counterfactual_eval.positives()),
                "counterfactual_global_value": None,
                "has_next_cycle": has_next,
                "canonical_next_global_value": cycles[i + 1].global_value.value
                if has_next
                else None,
                "canonical_next_positives": (
                    len(cycles[i + 1].evaluation.positives()) if has_next else None
                ),
                "is_terminal": not has_next,
            }
            rows.append(row)
            print(
                f"      cycle {i}: apex={positive_dim.value} surplus={surplus_dim.value} "
                f"counterfactual_positives={row['counterfactual_positives']} "
                f"canonical_next_positives={row['canonical_next_positives']}"
            )

    non_terminal = [r for r in rows if not r["is_terminal"]]
    wins = sum(
        1 for r in non_terminal if r["counterfactual_positives"] > r["canonical_next_positives"]
    )
    ties = sum(
        1 for r in non_terminal if r["counterfactual_positives"] == r["canonical_next_positives"]
    )
    losses = sum(
        1 for r in non_terminal if r["counterfactual_positives"] < r["canonical_next_positives"]
    )

    print()
    print(f"Refine cycles tested: {len(rows)} ({len(rows) - len(non_terminal)} terminal, excluded)")
    print(f"Counterfactual (surplus) beats canonical: {wins}")
    print(f"Counterfactual ties canonical:            {ties}")
    print(f"Counterfactual loses to canonical:         {losses}")

    out = {
        "dataset": args.dataset,
        "stories": len(stories),
        "max_cycles": args.max_cycles,
        "engine_depth": args.engine_depth,
        "summary": {
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "terminal_excluded": len(rows) - len(non_terminal),
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

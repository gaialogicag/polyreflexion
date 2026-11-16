"""Is the perspective divergence substantive, or is it framing vocabulary?

Each perspective prompt pushes its own register: the objective one says
"in fact", the subjective one says "believes", the conceptual one says
"in general". Two perspectives can therefore share almost no words while
selecting exactly the same facts out of the input. The raw measure cannot
tell those apart, and on templated inputs (T7, word sorting) the register
is a large share of every output, which is exactly where the raw numbers
behaved oddly.

This restricts both sides of every comparison to tokens that appear in the
SOURCE TEXT. What survives is the input content each perspective chose to
carry. If the perspectives still diverge there, they are selecting different
material and irreducibility is substantive. If they converge there while the
raw number stays high, the divergence was mostly register.

Makes no model calls: reads the JSON already collected and reloads the
datasets locally for their source texts.
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
from pathlib import Path

from polyrx.benchmark.runner import BenchmarkConfig, load_dataset_items
from polyrx.cli.compose import load_config

from perspective_divergence import content_tokens


def source_texts(dataset: str, stories: int) -> dict[str, str]:
    cfg = load_config([f"experiment={dataset}", "backends=openai_nano"])
    bench_cfg = BenchmarkConfig(
        dataset=cfg.dataset,
        num_items=stories,
        seed=cfg.experiment.seed,
        max_workers=cfg.experiment.max_workers,
        conditions=("answerer_direct",),
        prompts=cfg.prompts,
        paths=cfg.paths,
        report=cfg.report,
        postprocess=cfg.postprocess,
    )
    items, _ = load_dataset_items(bench_cfg)
    return {item.item_id: item.context for item in items}


def jaccard_distance(a: set[str], b: set[str]) -> float | None:
    if not a or not b:
        return None  # nothing grounded on one side; undefined, not zero
    return 1.0 - len(a & b) / len(a | b)


def summarise(name: str, xs: list[float]) -> dict:
    return {
        "name": name,
        "n": len(xs),
        "mean": statistics.mean(xs) if xs else float("nan"),
        "median": statistics.median(xs) if xs else float("nan"),
        "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="JSON from perspective_divergence.py")
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    collected: dict[str, list[dict[str, str]]] = data["per_story_perspectives"]
    texts = source_texts(args.dataset, data["stories"])

    missing = [sid for sid in collected if sid not in texts]
    if missing:
        print(f"warning: {len(missing)} story ids not found in dataset reload")

    # grounded[sid][leaf][perspective] = tokens of that output that occur in the source
    grounded: dict[str, list[dict[str, set[str]]]] = {}
    coverage: list[float] = []
    for sid, per_leaf in collected.items():
        if sid not in texts:
            continue
        src = content_tokens(texts[sid])
        rows = []
        for leaf in per_leaf:
            row = {}
            for p, out in leaf.items():
                toks = content_tokens(out)
                row[p] = toks & src
                if toks:
                    coverage.append(len(toks & src) / len(toks))
            rows.append(row)
        grounded[sid] = rows

    perspectives = sorted({p for per in grounded.values() for leaf in per for p in leaf})

    d_persp = [
        d
        for per_leaf in grounded.values()
        for leaf in per_leaf
        for a, b in itertools.combinations(perspectives, 2)
        if (d := jaccard_distance(leaf[a], leaf[b])) is not None
    ]
    d_region = [
        d
        for per_leaf in grounded.values()
        for p in perspectives
        for i, j in itertools.combinations(range(len(per_leaf)), 2)
        if (d := jaccard_distance(per_leaf[i][p], per_leaf[j][p])) is not None
    ]
    sids = list(grounded)
    d_item = [
        d
        for s1, s2 in itertools.combinations(sids, 2)
        for i in range(min(len(grounded[s1]), len(grounded[s2])))
        for p in perspectives
        if (d := jaccard_distance(grounded[s1][i][p], grounded[s2][i][p])) is not None
    ]

    rows = [
        summarise("grounded d_persp", d_persp),
        summarise("grounded d_region", d_region),
        summarise("grounded d_item", d_item),
    ]

    print(f"\n{args.dataset}: share of each output's words that come from the source: "
          f"{statistics.mean(coverage):.1%}")
    print(f"{'quantity':<26} {'n':>6} {'mean':>7} {'median':>7} {'sd':>6}")
    for r in rows:
        print(
            f"{r['name']:<26} {r['n']:>6} {r['mean']:>7.3f} "
            f"{r['median']:>7.3f} {r['stdev']:>6.3f}"
        )

    out = args.source.replace(".json", "_grounded.json")
    Path(out).write_text(
        json.dumps(
            {"dataset": args.dataset, "input_word_share": statistics.mean(coverage),
             "summary": rows},
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

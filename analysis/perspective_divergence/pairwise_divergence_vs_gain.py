"""The three pairwise divergences (O-S, O-B, S-B), correlated against gain
separately instead of averaged into one d_persp number.

divergence_vs_gain.py averages all three pairs per leaf before correlating.
If only one pair actually carries the controller's correction behavior, that
average can cancel out a real signal. This reads the same collected JSON and
reports each pair on its own, no new API calls.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from divergence_vs_gain import load_gain
from perspective_divergence import content_tokens, divergence

from polyrx.benchmark.runner import BenchmarkConfig, load_dataset_items
from polyrx.cli.compose import load_config

PAIRS = [("O", "S"), ("O", "B"), ("S", "B")]


def grounded_pair_divergence(leaves: list[dict[str, str]], a: str, b: str, source: str) -> float:
    src = content_tokens(source)
    vals = []
    for leaf in leaves:
        ta = content_tokens(leaf[a]) & src
        tb = content_tokens(leaf[b]) & src
        if ta or tb:
            vals.append(1.0 - len(ta & tb) / len(ta | tb))
    return statistics.mean(vals) if vals else float("nan")


def raw_pair_divergence(leaves: list[dict[str, str]], a: str, b: str) -> float:
    vals = [divergence(leaf[a], leaf[b]) for leaf in leaves]
    return statistics.mean(vals) if vals else float("nan")


def pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
    return num / den if den else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divergence", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gain", nargs="+", required=True)
    ap.add_argument("--direct-from", default=None)
    ap.add_argument("--by-context", action="store_true")
    args = ap.parse_args()

    div_data = json.loads(Path(args.divergence).read_text(encoding="utf-8"))
    collected: dict[str, list[dict[str, str]]] = div_data["per_story_perspectives"]
    gains = load_gain(args.gain, args.direct_from)

    cfg = load_config([f"experiment={args.dataset}", "backends=openai_nano"])
    bench = BenchmarkConfig(
        dataset=cfg.dataset,
        num_items=10**6,
        seed=cfg.experiment.seed,
        conditions=("answerer_direct",),
        prompts=cfg.prompts,
        paths=cfg.paths,
        report=cfg.report,
        postprocess=cfg.postprocess,
    )
    items, _ = load_dataset_items(bench)
    by_id = {i.item_id: i for i in items}
    ctx_members: dict[str, list[str]] = {}
    for i in items:
        ctx_members.setdefault(i.context, []).append(i.item_id)

    rows = []
    for sid, leaves in collected.items():
        item = by_id.get(sid)
        if item is None:
            continue
        members = ctx_members[item.context] if args.by_context else [sid]
        obs = [gains[m] for m in members if m in gains]
        if not obs:
            continue
        row = {
            "sid": sid,
            "n_questions": len(obs),
            "direct_rate": sum(d for d, _ in obs) / len(obs),
            "meta_rate": sum(m for _, m in obs) / len(obs),
        }
        for a, b in PAIRS:
            row[f"grounded_{a}{b}"] = grounded_pair_divergence(leaves, a, b, item.context)
            row[f"raw_{a}{b}"] = raw_pair_divergence(leaves, a, b)
        rows.append(row)

    print(
        f"\n{args.dataset}: {len(rows)} independent units "
        f"({sum(r['n_questions'] for r in rows)} graded questions)\n"
    )

    ys = (
        [r["meta_rate"] - r["direct_rate"] for r in rows]
        if args.by_context
        else [int(r["meta_rate"] >= 0.5) - int(r["direct_rate"] >= 0.5) for r in rows]
    )
    kind = "Pearson" if args.by_context else "point-biserial"

    print(f"{'pair':<10} {'metric':<10} {'mean':>7} {kind + ' r':>16}")
    for a, b in PAIRS:
        for metric_prefix in ("grounded", "raw"):
            key = f"{metric_prefix}_{a}{b}"
            xs = [r[key] for r in rows]
            r = pearson(xs, ys)
            print(f"{a}-{b:<8} {metric_prefix:<10} {statistics.mean(xs):>7.3f} {r:>+16.3f}")
        print()

    # also: does the SPREAD across the three pairs (not their average) predict gain?
    # a story where the three disagree unevenly (one pair far more divergent than
    # the others) is a different shape of disagreement than one where all three
    # pairs diverge equally -- the plain average cannot distinguish them.
    print("--- spread across the 3 pairs (stdev of O-S, O-B, S-B per story) ---")
    for metric_prefix in ("grounded", "raw"):
        spreads = [
            statistics.pstdev([r[f"{metric_prefix}_{a}{b}"] for a, b in PAIRS]) for r in rows
        ]
        r = pearson(spreads, ys)
        print(
            f"{metric_prefix:<10} mean spread {statistics.mean(spreads):.3f}  "
            f"{kind} r(spread, gain) = {r:+.3f}"
        )


if __name__ == "__main__":
    main()

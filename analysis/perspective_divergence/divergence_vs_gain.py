"""Does perspective divergence predict where the architecture helps?

Experiment 3 showed the three contextures select different material from the
input. That is a property of the architecture, measured in isolation. This
asks the question that would make it useful: on an item where the three
perspectives diverge more, does the meta-steered pipeline beat direct
answering by more?

Outcomes here are binary per item (each condition is right or wrong), so a
plain correlation is the wrong tool. Two things are reported instead:

  cells     mean divergence within each of the four outcome cells
            (both right / both wrong / meta-only right / direct-only right).
            If divergence carries signal, meta-only-right and
            direct-only-right should separate.

  logistic  P(meta_c4 correct) as a function of divergence, fitted
            separately among items direct got right and items direct got
            wrong. Splitting on the direct outcome keeps item difficulty
            from doing the work that divergence is being credited for.

Datasets whose items share a context (Hi-ToM) are aggregated to the context:
every question on one story shares that story's reflexion tree, so the
questions are not independent observations of divergence.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from perspective_divergence import content_tokens, divergence

from polyrx.benchmark.runner import BenchmarkConfig, load_dataset_items
from polyrx.cli.compose import load_config


def load_gain(paths: list[str], direct_from: str | None) -> dict[str, tuple[bool, bool]]:
    """item_id -> (direct_correct, meta_c4_correct), merged across runs."""
    direct: dict[str, bool] = {}
    meta: dict[str, bool] = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for r in d["results"]:
            j = r.get("judgments", {})
            if "answerer_meta_c4" in j:
                meta[r["item_id"]] = bool(j["answerer_meta_c4"].get("correct"))
            if "answerer_direct" in j and direct_from is None:
                direct[r["item_id"]] = bool(j["answerer_direct"].get("correct"))
    if direct_from is not None:
        d = json.loads(Path(direct_from).read_text(encoding="utf-8"))
        for r in d["results"]:
            j = r.get("judgments", {})
            if "answerer_direct" in j:
                direct[r["item_id"]] = bool(j["answerer_direct"].get("correct"))
    return {k: (direct[k], meta[k]) for k in direct.keys() & meta.keys()}


def grounded_persp_divergence(leaves: list[dict[str, str]], source: str) -> float:
    """Mean pairwise O/S/B divergence within leaves, over input-grounded words."""
    src = content_tokens(source)
    vals = []
    for leaf in leaves:
        toks = {p: content_tokens(t) & src for p, t in leaf.items()}
        for a, b in itertools.combinations(sorted(toks), 2):
            ta, tb = toks[a], toks[b]
            if ta or tb:
                vals.append(1.0 - len(ta & tb) / len(ta | tb))
    return statistics.mean(vals) if vals else float("nan")


def raw_persp_divergence(leaves: list[dict[str, str]]) -> float:
    vals = [
        divergence(leaf[a], leaf[b])
        for leaf in leaves
        for a, b in itertools.combinations(sorted(leaf), 2)
    ]
    return statistics.mean(vals) if vals else float("nan")


def logistic_fit(xs: list[float], ys: list[int], iters: int = 2000, lr: float = 0.5):
    """Tiny gradient-descent logistic regression; returns (intercept, slope)."""
    if len(set(ys)) < 2:
        return None
    mx = statistics.mean(xs)
    sx = statistics.pstdev(xs) or 1.0
    zs = [(x - mx) / sx for x in xs]
    b0, b1 = 0.0, 0.0
    n = len(zs)
    for _ in range(iters):
        g0 = g1 = 0.0
        for z, y in zip(zs, ys, strict=True):
            p = 1 / (1 + math.exp(-(b0 + b1 * z)))
            g0 += p - y
            g1 += (p - y) * z
        b0 -= lr * g0 / n
        b1 -= lr * g1 / n
    return b0, b1 / sx  # slope back in original units


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--divergence", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gain", nargs="+", required=True)
    ap.add_argument(
        "--direct-from",
        default=None,
        help="take direct results from this run instead (post-fix baseline)",
    )
    ap.add_argument(
        "--by-context", action="store_true", help="aggregate items to their shared context (Hi-ToM)"
    )
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
        rows.append(
            {
                "sid": sid,
                "d_raw": raw_persp_divergence(leaves),
                "d_grounded": grounded_persp_divergence(leaves, item.context),
                "n_questions": len(obs),
                "direct_rate": sum(d for d, _ in obs) / len(obs),
                "meta_rate": sum(m for _, m in obs) / len(obs),
            }
        )

    print(
        f"\n{args.dataset}: {len(rows)} independent units "
        f"({sum(r['n_questions'] for r in rows)} graded questions)"
    )

    for metric in ("d_raw", "d_grounded"):
        print(f"\n--- {metric} ---")
        # Pearson between divergence and gain. By-context: gain is a rate
        # (meta_rate - direct_rate) over multiple questions per story, so this
        # is an ordinary Pearson correlation. Per-item: gain is binary (meta
        # right minus direct right, in {-1, 0, +1}), so this is the
        # point-biserial correlation -- same formula, valid either way.
        xs = [r[metric] for r in rows]
        ys = (
            [r["meta_rate"] - r["direct_rate"] for r in rows]
            if args.by_context
            else [int(r["meta_rate"] >= 0.5) - int(r["direct_rate"] >= 0.5) for r in rows]
        )
        mx, my = statistics.mean(xs), statistics.mean(ys)
        num = sum((a - mx) * (b - my) for a, b in zip(xs, ys, strict=True))
        den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys))
        r = num / den if den else float("nan")
        kind = "Pearson" if args.by_context else "point-biserial"
        print(f"  mean gain {my:+.3f}; {kind} r(divergence, gain) = {r:+.3f} over n={len(xs)}")
        cells = {"both right": [], "both wrong": [], "meta only": [], "direct only": []}
        for r_ in rows:
            d, m = r_["direct_rate"] >= 0.5, r_["meta_rate"] >= 0.5
            key = (
                "both right"
                if d and m
                else "both wrong"
                if not d and not m
                else "meta only"
                if m
                else "direct only"
            )
            cells[key].append(r_[metric])
        for k, v in cells.items():
            if v:
                print(f"  {k:<12} n={len(v):4d} mean {statistics.mean(v):.3f}")
        split = {True: ([], []), False: ([], [])}
        for r_ in rows:
            d = r_["direct_rate"] >= 0.5
            split[d][0].append(r_[metric])
            split[d][1].append(1 if r_["meta_rate"] >= 0.5 else 0)
        for d, (xs, ys) in split.items():
            if len(xs) > 10:
                fit = logistic_fit(xs, ys)
                label = "direct right" if d else "direct wrong"
                if fit:
                    print(
                        f"  logistic P(meta right | {label}): slope {fit[1]:+.2f} per unit divergence, n={len(xs)}"
                    )
                else:
                    print(
                        f"  logistic P(meta right | {label}): degenerate (all same outcome), n={len(xs)}"
                    )


if __name__ == "__main__":
    main()

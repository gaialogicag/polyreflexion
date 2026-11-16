"""Do the three contextures actually diverge, or do they collapse?

Günther's premise is that the three perspectives are mutually irreducible.
That is assumed in the architecture, never measured. This script measures it
directly, on real reflexion trees, against two reference points:

  d_persp   O vs S vs B on identical text AND identical region context.
            This is the quantity the premise is about.
  d_region  the SAME perspective across the three depth-1 leaves. Those leaves
            differ only in region/apex context, not in contexture. This is the
            control: if d_persp is no larger than d_region, then "a different
            contexture" is doing no more work than "a slightly different
            prompt context", and the three-ness is decorative.
  d_item    the same perspective on DIFFERENT stories. Ceiling: unrelated text.

Run with the project's own config tree, e.g.

    python perspective_divergence.py --dataset exploretom --stories 30
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from pathlib import Path

from polyrx.benchmark.runner import BenchmarkConfig, load_dataset_items
from polyrx.cli.compose import load_config
from polyrx.engine import Node, Perspective, PromptRegistry, ReflexionEngine
from polyrx.models.registry import get_client, set_active_backends

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "in", "is", "it", "its",
    "of", "on", "or", "she", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "were", "what", "when", "which",
    "who", "will", "with", "would", "you", "your", "not", "no", "do", "does",
    "did", "can", "could", "should", "if", "about", "into", "than", "so",
}


def content_tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 2}


def divergence(a: str, b: str) -> float:
    """1 - Jaccard over content words. 0 = identical wording, 1 = disjoint."""
    ta, tb = content_tokens(a), content_tokens(b)
    if not ta and not tb:
        return 0.0
    return 1.0 - len(ta & tb) / len(ta | tb)


def leaves(node: Node) -> list[Node]:
    if node.is_leaf:
        return [node]
    assert node.children is not None
    return [leaf for child in node.children for leaf in leaves(child)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="exploretom")
    ap.add_argument("--stories", type=int, default=30)
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--out", default="perspective_divergence.json")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    overrides = [f"experiment={args.dataset}", "backends=openai_nano", *args.overrides]
    cfg = load_config(overrides)
    set_active_backends(cfg.backends, cfg.postprocess)
    prompts = PromptRegistry(cfg.prompts.reflexion)
    client = get_client("answerer")

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

    # One reflexion tree per distinct story, not per question: the engine runs
    # on the context, and every question on that story shares the same tree.
    seen: dict[str, str] = {}
    for item in items:
        if item.context not in seen.values():
            seen[item.item_id] = item.context
        if len(seen) >= args.stories:
            break
    stories = list(seen.items())
    print(f"{len(stories)} distinct stories from {args.dataset}")

    # story_id -> leaf_index -> perspective -> text
    collected: dict[str, list[dict[str, str]]] = {}
    for n, (sid, text) in enumerate(stories, 1):
        with ReflexionEngine(client, prompts=prompts, max_depth=args.depth) as engine:
            result = engine.run(text)
        per_leaf = []
        for leaf in leaves(result.tree):
            assert leaf.perspectives is not None
            per_leaf.append({p.value: leaf.perspectives[p] for p in Perspective})
        collected[sid] = per_leaf
        print(f"  [{n}/{len(stories)}] {sid}: {len(per_leaf)} leaves")

    # d_persp: pairwise across the three contextures, within one leaf.
    d_persp = [
        divergence(leaf[a], leaf[b])
        for per_leaf in collected.values()
        for leaf in per_leaf
        for a, b in itertools.combinations(sorted(leaf), 2)
    ]

    # d_region: same contexture, different leaf (differs only by region context).
    d_region = [
        divergence(per_leaf[i][p], per_leaf[j][p])
        for per_leaf in collected.values()
        for p in [pp.value for pp in Perspective]
        for i, j in itertools.combinations(range(len(per_leaf)), 2)
    ]

    # d_item: same contexture, same leaf index, different story. Ceiling.
    sids = list(collected)
    d_item = [
        divergence(collected[s1][i][p], collected[s2][i][p])
        for s1, s2 in itertools.combinations(sids, 2)
        for i in range(min(len(collected[s1]), len(collected[s2])))
        for p in [pp.value for pp in Perspective]
    ]

    def describe(name: str, xs: list[float]) -> dict:
        return {
            "name": name,
            "n": len(xs),
            "mean": statistics.mean(xs),
            "median": statistics.median(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min": min(xs),
            "max": max(xs),
        }

    rows = [
        describe("d_persp (O/S/B, same text+region)", d_persp),
        describe("d_region (same perspective, other leaf)", d_region),
        describe("d_item (same perspective, other story)", d_item),
    ]

    print()
    print(f"{'quantity':<42} {'n':>6} {'mean':>7} {'median':>7} {'sd':>6}")
    for r in rows:
        print(
            f"{r['name']:<42} {r['n']:>6} {r['mean']:>7.3f} "
            f"{r['median']:>7.3f} {r['stdev']:>6.3f}"
        )

    out = {
        "dataset": args.dataset,
        "depth": args.depth,
        "stories": len(stories),
        "summary": rows,
        "per_story_perspectives": collected,
    }
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

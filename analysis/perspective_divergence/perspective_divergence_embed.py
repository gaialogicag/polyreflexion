"""Second opinion on perspective divergence, using embeddings instead of words.

The lexical measure cannot separate two things that matter here: perspectives
that say different things, and perspectives that say similar things in the
vocabulary their own prompt pushed them toward ("believes", "in fact",
"in general"). Cosine distance over embeddings is not immune to that either,
but it fails differently, so agreement between the two is worth more than
either alone.

Reads the JSON written by perspective_divergence.py; makes no model calls
beyond embedding the texts already collected.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
from pathlib import Path

from openai import OpenAI


def embed_all(texts: list[str], model: str, batch: int = 128) -> list[list[float]]:
    client = OpenAI()
    out: list[list[float]] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        resp = client.embeddings.create(model=model, input=chunk)
        out.extend(d.embedding for d in resp.data)
        print(f"  embedded {min(i + batch, len(texts))}/{len(texts)}")
    return out


def cosine_distance(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / (na * nb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="JSON from perspective_divergence.py")
    ap.add_argument("--model", default="text-embedding-3-small")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(Path(args.source).read_text(encoding="utf-8"))
    collected: dict[str, list[dict[str, str]]] = data["per_story_perspectives"]

    index: dict[tuple[str, int, str], int] = {}
    texts: list[str] = []
    for sid, per_leaf in collected.items():
        for li, leaf in enumerate(per_leaf):
            for p, text in leaf.items():
                index[(sid, li, p)] = len(texts)
                texts.append(text)

    print(f"embedding {len(texts)} perspective texts with {args.model}")
    vecs = embed_all(texts, args.model)

    def dist(k1: tuple[str, int, str], k2: tuple[str, int, str]) -> float:
        return cosine_distance(vecs[index[k1]], vecs[index[k2]])

    perspectives = sorted({p for per in collected.values() for leaf in per for p in leaf})

    d_persp = [
        dist((sid, li, a), (sid, li, b))
        for sid, per_leaf in collected.items()
        for li in range(len(per_leaf))
        for a, b in itertools.combinations(perspectives, 2)
    ]
    d_region = [
        dist((sid, i, p), (sid, j, p))
        for sid, per_leaf in collected.items()
        for p in perspectives
        for i, j in itertools.combinations(range(len(per_leaf)), 2)
    ]
    sids = list(collected)
    d_item = [
        dist((s1, i, p), (s2, i, p))
        for s1, s2 in itertools.combinations(sids, 2)
        for i in range(min(len(collected[s1]), len(collected[s2])))
        for p in perspectives
    ]

    rows = []
    for name, xs in [
        ("d_persp (O/S/B, same text+region)", d_persp),
        ("d_region (same perspective, other leaf)", d_region),
        ("d_item (same perspective, other story)", d_item),
    ]:
        rows.append(
            {
                "name": name,
                "n": len(xs),
                "mean": statistics.mean(xs),
                "median": statistics.median(xs),
                "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            }
        )

    print()
    print(f"{'quantity':<42} {'n':>6} {'mean':>7} {'median':>7} {'sd':>6}")
    for r in rows:
        print(
            f"{r['name']:<42} {r['n']:>6} {r['mean']:>7.3f} {r['median']:>7.3f} {r['stdev']:>6.3f}"
        )

    out = args.out or args.source.replace(".json", "_embed.json")
    Path(out).write_text(
        json.dumps({"model": args.model, "summary": rows}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

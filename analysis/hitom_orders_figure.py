"""Hi-ToM accuracy by belief order: direct, depth 1, depth 1 + 4 meta cycles, for two models.

Reads the saved per-question results, writes a single-column PDF for the paper, one panel per model, stacked.
Usage: python analysis/hitom_orders_figure.py <output.pdf>
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = {
    "gpt-4.1-nano": {
        "$T^0$": ("hitom_20261015_104654", "answerer_direct"),
        "$T^1$": ("hitom_20261013_075538", "answerer_reflexion"),
        "$T^1, c=4$": ("hitom_20261006_225845", "answerer_meta_c4"),
    },
    "Qwen2.5-3b": {
        "$T^0$": ("hitom_20261021_175841", "answerer_direct"),
        "$T^1$": ("hitom_20261021_210621", "answerer_reflexion"),
        "$T^1, c=4$": ("hitom_20261022_220753", "answerer_meta_c4"),
    },
}
# Categorical slots 1-3 of the validated reference palette (light surface).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
ORDERS = [f"order {k}" for k in range(5)]


def load(name):
    path = f"results/raw/hitom/{name}.json"
    return {r["item_id"]: r for r in json.load(open(path))["results"]}


def accuracy_by_order(runs):
    data = {label: load(name) for label, (name, _) in runs.items()}
    ids = sorted(set.intersection(*(set(d) for d in data.values())))
    group = {i: next(iter(data.values()))[i]["group"] for i in ids}
    acc = {}
    for label, (_, cond) in runs.items():
        acc[label] = []
        for order in ORDERS:
            sel = [i for i in ids if group[i] == order]
            correct = sum(bool(data[label][i]["judgments"][cond]["correct"]) for i in sel)
            acc[label].append(100 * correct / len(sel))
    return acc, len(ids)


plt.rcParams.update({"font.family": "serif", "font.size": 7, "axes.linewidth": 0.6})
fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.35), dpi=300, sharex=True, sharey=True)
width = 0.3
for ax, (model, runs) in zip(axes, RUNS.items(), strict=True):
    acc, n = accuracy_by_order(runs)
    for k, (label, values) in enumerate(acc.items()):
        xs = [x + (k - 1) * width for x in range(5)]
        ax.bar(xs, values, width=width * 0.88, color=COLORS[k], label=label, linewidth=0)
        for x, v in zip(xs, values, strict=True):
            ax.text(x, v + 0.8, f"{v:.1f}", ha="center", va="bottom", fontsize=4.4, color="#52514e")
    ax.set_title(model, fontsize=7.5, loc="left")
    ax.set_xticks(range(5))
    ax.tick_params(labelbottom=True)
    ax.set_xticklabels([str(k) for k in range(5)])
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 62)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    print(model, {k: [round(v, 1) for v in vs] for k, vs in acc.items()}, "questions:", n)
axes[-1].set_xlabel("belief order")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    frameon=False,
    ncol=3,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.04),
    handlelength=1.0,
    columnspacing=1.6,
)
fig.tight_layout()
fig.savefig(sys.argv[1], bbox_inches="tight")
fig.savefig(sys.argv[1].replace(".pdf", ".png"), bbox_inches="tight", dpi=200)

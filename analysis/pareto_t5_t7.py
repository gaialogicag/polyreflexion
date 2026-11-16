"""Cost/accuracy Pareto frontier for T5 and T7.

One point per direct-answering model (T^0), plus gpt-4.1-nano's depth-1
reflexion condition (T^1) as the architecture's entry on the same axes.
Costs are the measured API spend for the full 250-item task.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, accuracy %, cost $, family)
T5 = [
    ("gpt-4.1-nano", 17.6, 0.0209, "openai"),
    ("gpt-4.1", 36.0, 0.2233, "openai"),
    ("gpt-4o", 34.0, 0.2791, "openai"),
    ("gpt-5.1", 35.6, 0.1680, "openai"),
    ("gpt-5-mini", 100.0, 0.2527, "openai"),
    ("gpt-5", 100.0, 1.1888, "openai"),
    ("gemini-2.5-flash-lite", 100.0, 0.0717, "gemini"),
    ("gemini-2.5-flash", 94.8, 0.3203, "gemini"),
    ("gemini-3.5-flash-lite", 66.8, 0.1255, "gemini"),
    ("gemini-3.8-flash", 99.2, 0.2208, "gemini"),
    ("gemini-2.5-pro", 100.0, 2.3456, "gemini"),
]
T5_ARCH = ("gpt-4.1-nano $T^1$", 91.6, 0.3690)

T7 = [
    ("gpt-4.1-nano", 17.6, 0.0141, "openai"),
    ("gpt-4.1", 35.6, 0.2753, "openai"),
    ("gpt-4o", 39.6, 0.3441, "openai"),
    ("gpt-5.1", 38.0, 0.2005, "openai"),
    ("gpt-5-mini", 100.0, 0.3400, "openai"),
    ("gpt-5", 100.0, 1.9304, "openai"),
]
T7_ARCH = ("gpt-4.1-nano $T^1$", 85.6, 0.4711)

COLOR = {"openai": "#4C72B0", "gemini": "#DD8452"}
ARCH_COLOR = "#C44E52"


def frontier(points):
    """Points not dominated: nothing cheaper is at least as accurate."""
    out, best = [], -1.0
    for _label, acc, cost, *_ in sorted(points, key=lambda p: p[2]):
        if acc > best:
            out.append((cost, acc))
            best = acc
    return out


OFFSET = {
    "gpt-5-mini": (6, -11),
    "gemini-3.8-flash": (-46, 5),
    "gemini-2.5-flash": (-44, -12),
    "gemini-2.5-flash-lite": (-18, 7),
    "gpt-5": (6, 2),
    "gemini-2.5-pro": (-30, 7),
    "gpt-4o": (5, -2),
    "gpt-4.1": (-2, 7),
    "gpt-5.1": (-24, 2),
    "gpt-4.1-nano": (6, -3),
}


def panel(ax, models, arch, title):
    pts = [(lab, acc, cost) for lab, acc, cost, _ in models] + [arch]
    front = frontier([(p[0], p[1], p[2]) for p in pts])
    ax.step(
        [c for c, _ in front],
        [a for _, a in front],
        where="post",
        color="0.55",
        lw=1.2,
        zorder=1,
        label="Pareto frontier",
    )

    for _lab, acc, cost, fam in models:
        ax.scatter(cost, acc, s=46, color=COLOR[fam], zorder=3, edgecolor="white", linewidth=0.6)
    lab, acc, cost = arch
    ax.scatter(
        cost, acc, s=150, marker="*", color=ARCH_COLOR, zorder=4, edgecolor="white", linewidth=0.7
    )

    for lab, acc, cost, *_ in [(a, b, c, d) for a, b, c, d in models] + [(*arch, "arch")]:
        short = lab.replace("gemini-", "g-").replace("gpt-", "")
        off = (8, -4) if lab.endswith("$T^1$") else OFFSET.get(lab, (5, 4))
        ax.annotate(
            short,
            (cost, acc),
            textcoords="offset points",
            xytext=off,
            fontsize=6.5,
            color=ARCH_COLOR if lab.endswith("$T^1$") else "0.25",
        )

    ax.set_xscale("log")
    ax.set_xlabel("API cost for 250 items (USD, log scale)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(5, 112)


fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
panel(axes[0], T5, T5_ARCH, "T5 — tracking shuffled objects, five objects")
panel(axes[1], T7, T7_ARCH, "T7 — tracking shuffled objects, seven objects")
axes[0].set_ylabel("Accuracy (%)")

handles = [
    plt.Line2D([], [], marker="o", ls="", color=COLOR["openai"], label="OpenAI $T^0$"),
    plt.Line2D([], [], marker="o", ls="", color=COLOR["gemini"], label="Gemini $T^0$"),
    plt.Line2D(
        [], [], marker="*", ls="", ms=11, color=ARCH_COLOR, label="gpt-4.1-nano $T^1$ (ours)"
    ),
    plt.Line2D([], [], color="0.55", lw=1.2, label="Pareto frontier"),
]
axes[1].legend(handles=handles, fontsize=7.5, loc="center right", framealpha=0.95)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{__import__('sys').argv[1]}/pareto_t5_t7.{ext}", dpi=200, bbox_inches="tight")
print("written")

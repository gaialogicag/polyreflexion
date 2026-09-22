"""Sierpinski topology rendering for meta runs.

Per cycle one PNG shows the current triangle, the canonical move (shaded) and
the surplus arrangement (dashed outline — preserved, never used).  A combined
"evolution" figure overlays all cycles with increasing color depth.

Geometry model: the renderer tracks the *current* triangle as a mapping
``Perspective -> (x, y)``.  A descent replaces it with a corner sub-triangle,
an expansion mirrors it across the anchored edge, a reset returns to the root.
"""

from __future__ import annotations

from pathlib import Path

from polyreflexion.charts import pyplot
from polyreflexion.engine import Perspective
from polyreflexion.meta.datatypes import GeometricArrangement, MetaResult

Point = tuple[float, float]
Triangle = dict[Perspective, Point]

# Root triangle: O on top, S bottom-left, B bottom-right.
ROOT: Triangle = {
    Perspective.OBJECTIVITY: (0.5, 3**0.5 / 2),
    Perspective.SUBJECTIVITY: (0.0, 0.0),
    Perspective.CONCEPTUALITY: (1.0, 0.0),
}
_DRAW_ORDER = (Perspective.OBJECTIVITY, Perspective.SUBJECTIVITY, Perspective.CONCEPTUALITY)


def _midpoint(a: Point, b: Point) -> Point:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _reflect(point: Point, a: Point, b: Point) -> Point:
    """Reflect ``point`` across the line through ``a`` and ``b``."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    denom = dx * dx + dy * dy
    t = ((point[0] - a[0]) * dx + (point[1] - a[1]) * dy) / denom
    foot = (a[0] + t * dx, a[1] + t * dy)
    return (2 * foot[0] - point[0], 2 * foot[1] - point[1])


def _corner_subtriangle(triangle: Triangle, apex: Perspective) -> Triangle:
    """Corner sub-triangle at ``apex``: apex stays, the others become midpoints."""
    sub: Triangle = {apex: triangle[apex]}
    for vertex, point in triangle.items():
        if vertex is not apex:
            sub[vertex] = _midpoint(triangle[apex], point)
    return sub


def _mirror_triangle(triangle: Triangle, edge: tuple[Perspective, Perspective]) -> Triangle:
    """Expansion: mirror the triangle outward across the anchored edge."""
    (third,) = [v for v in triangle if v not in edge]
    mirrored = dict(triangle)
    mirrored[third] = _reflect(triangle[third], triangle[edge[0]], triangle[edge[1]])
    return mirrored


def _apply(triangle: Triangle, arrangement: GeometricArrangement) -> tuple[Triangle, Triangle]:
    """Return ``(shape to draw, new current triangle)`` for one arrangement."""
    kind = arrangement.kind
    if kind == "descend_corner":
        sub = _corner_subtriangle(triangle, arrangement.anchor[0])
        return sub, sub
    if kind == "expand_edge":
        mirrored = _mirror_triangle(triangle, (arrangement.anchor[0], arrangement.anchor[1]))
        return mirrored, mirrored
    if kind in ("rest", "expand_full"):
        return triangle, triangle
    if kind in ("reset_root", "invert_root"):
        return dict(ROOT), dict(ROOT)
    return triangle, triangle  # unknown kinds render in place


def _coords(triangle: Triangle) -> list[Point]:
    return [triangle[v] for v in _DRAW_ORDER]


def _draw_outline(ax, triangle: Triangle, **kwargs) -> None:
    xs = [p[0] for p in _coords(triangle)] + [_coords(triangle)[0][0]]
    ys = [p[1] for p in _coords(triangle)] + [_coords(triangle)[0][1]]
    ax.plot(xs, ys, **kwargs)


def _draw_filled(ax, triangle: Triangle, **kwargs) -> None:
    xs = [p[0] for p in _coords(triangle)]
    ys = [p[1] for p in _coords(triangle)]
    ax.fill(xs, ys, **kwargs)


def _setup_axes(ax, title: str) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 2.2)
    ax.set_ylim(-1.2, 2.0)
    ax.axis("off")
    ax.set_title(title, fontsize=10)
    # Root outline plus vertex labels for orientation.
    _draw_outline(ax, ROOT, color="black", linewidth=1.2)
    for vertex, (x, y) in ROOT.items():
        offset = 0.07 if vertex is Perspective.OBJECTIVITY else -0.11
        ax.text(x, y + offset, vertex.value, ha="center", fontsize=11, weight="bold")


def render_topology(result: MetaResult, out_dir: Path | str) -> list[Path]:
    """Render one PNG per cycle plus a combined evolution figure.

    Returns the list of written file paths (cycle images first), or an empty
    list when matplotlib is not installed — the trace and report are still
    written, only the pictures are skipped.
    """
    plt = pyplot()
    if plt is None:
        print("Topology renderings skipped: install polyreflexion[viz] for charts.")
        return []
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Pre-compute every cycle's drawn shape by walking the moves in order.
    current: Triangle = dict(ROOT)
    steps: list[tuple[int, Triangle, Triangle, str]] = []  # (idx, canonical, surplus, label)
    for cycle in result.cycles:
        canonical_shape, next_current = _apply(current, cycle.topology.canonical)
        surplus_shape, _ = _apply(current, cycle.topology.surplus)
        label = (
            f"cycle {cycle.index}: {cycle.evaluation.as_triple()} → "
            f"{cycle.global_value.value} ({cycle.topology.canonical.kind})"
        )
        steps.append((cycle.index, canonical_shape, surplus_shape, label))
        current = next_current

    cmap = plt.get_cmap("Blues")

    # One figure per cycle: history light, current move shaded, surplus dashed.
    for position, (index, canonical, surplus, label) in enumerate(steps):
        fig, ax = plt.subplots(figsize=(5, 4.5))
        _setup_axes(ax, label)
        for _, prior_canonical, _, _ in steps[:position]:
            _draw_filled(ax, prior_canonical, color="lightsteelblue", alpha=0.35)
        _draw_filled(ax, canonical, color=cmap(0.65), alpha=0.75)
        _draw_outline(ax, surplus, color="crimson", linestyle="--", linewidth=1.3, alpha=0.9)
        path = out_dir / f"topology_cycle{index}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # Combined evolution figure: all canonical moves, deeper color = later cycle.
    fig, ax = plt.subplots(figsize=(6, 5.5))
    _setup_axes(ax, "Topology evolution (canonical moves; dashed = last surplus)")
    total = max(len(steps), 1)
    for position, (index, canonical, surplus, _) in enumerate(steps):
        shade = 0.25 + 0.6 * (position + 1) / total
        _draw_filled(ax, canonical, color=cmap(shade), alpha=0.55, label=f"cycle {index}")
        if position == len(steps) - 1:
            _draw_outline(ax, surplus, color="crimson", linestyle="--", linewidth=1.3)
    if steps:
        ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    overview = out_dir / "topology_evolution.png"
    fig.savefig(overview, dpi=150, bbox_inches="tight")
    plt.close(fig)
    written.append(overview)
    return written

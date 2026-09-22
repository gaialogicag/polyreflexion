"""Deterministic mapping from a global valuation to Sierpinski geometry.

For every valuation there is exactly one canonical geometric interpretation
plus exactly one additional unused arrangement (the "semantic surplus").
The surplus is preserved in every ``TopologyState`` — it never drives control
flow in this version, but future meta-levels may consume it.

Vertex anchoring: subjective -> S, objective -> O, dialectical -> B, matching
the vertices of the existing reflexion triangle.
"""

from __future__ import annotations

from polyreflexion.engine import Perspective
from polyreflexion.meta.datatypes import (
    DIMENSION_TO_VERTEX,
    Dimension,
    GeometricArrangement,
    GlobalValue,
    TopologyState,
)

# Stable vertex order used for sorting anchors and rotating arrangements.
_VERTEX_ORDER: tuple[Perspective, ...] = (
    Perspective.OBJECTIVITY,
    Perspective.SUBJECTIVITY,
    Perspective.CONCEPTUALITY,
)
_ALL_VERTICES: tuple[Perspective, ...] = _VERTEX_ORDER


def _rotate(vertex: Perspective) -> Perspective:
    """Rotate one step in canonical vertex order (deterministic surplus rule)."""
    return _VERTEX_ORDER[(_VERTEX_ORDER.index(vertex) + 1) % 3]


def _sorted_vertices(dimensions) -> tuple[Perspective, ...]:
    vertices = [DIMENSION_TO_VERTEX[d] for d in dimensions]
    return tuple(sorted(vertices, key=_VERTEX_ORDER.index))


class GeometryMapper:
    """Translate (GlobalValue, positive dimensions) into a topology move."""

    def map(
        self,
        global_value: GlobalValue,
        positives: frozenset[Dimension],
        path: tuple[str, ...] = (),
    ) -> TopologyState:
        """Return the canonical + surplus arrangement and the updated descent path."""
        if global_value is GlobalValue.R:
            # Two positive dimensions identify one edge: expand across it.
            edge = _sorted_vertices(positives)
            canonical = GeometricArrangement("expand_edge", edge)
            # Surplus: the same expansion under the rotated vertex assignment.
            surplus = GeometricArrangement(
                "expand_edge",
                tuple(sorted((_rotate(v) for v in edge), key=_VERTEX_ORDER.index)),
            )
            new_path = (*path, "+" + "".join(v.value for v in edge))

        elif global_value is GlobalValue.A:
            # One positive dimension = apex; the two negative rationales become
            # the left/right boundaries of the corner sub-triangle.
            (positive,) = tuple(positives)
            apex = DIMENSION_TO_VERTEX[positive]
            canonical = GeometricArrangement("descend_corner", (apex,))
            negatives = _sorted_vertices(set(Dimension) - set(positives))
            surplus = GeometricArrangement("descend_corner", (negatives[0],))
            new_path = (*path, apex.value)

        elif global_value is GlobalValue.W:
            # Converged: no move. Surplus records the virtual full expansion.
            canonical = GeometricArrangement("rest", ())
            surplus = GeometricArrangement("expand_full", _ALL_VERTICES)
            new_path = path

        elif global_value is GlobalValue.F:
            # Total failure: back to the root triangle. Surplus is the inversion.
            canonical = GeometricArrangement("reset_root", ())
            surplus = GeometricArrangement("invert_root", _ALL_VERTICES)
            new_path = ()

        else:  # pragma: no cover - enum is exhaustive
            raise ValueError(f"Unknown global value: {global_value!r}")

        return TopologyState(canonical=canonical, surplus=surplus, path=new_path)

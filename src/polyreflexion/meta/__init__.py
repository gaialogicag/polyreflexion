"""Polycontextural meta-evaluation layer.

This package sits strictly above the recursive reflexion engine.  It never
reasons about content itself: it observes one completed engine run as a single
object, evaluates it from three logical contexts (subjective / objective /
dialectical), maps the valuation triple onto a deterministic Sierpinski
geometry, and organizes the next recursive cycle (expansion, refinement, reset,
or termination).  All reasoning text is produced by the engine; the meta layer
only produces structure.
"""

from polyreflexion.meta.datatypes import (
    BoundaryStatement,
    ContextJudgment,
    Decision,
    Dimension,
    GeometricArrangement,
    GlobalValue,
    MetaConfig,
    MetaCycle,
    MetaResult,
    Observation,
    PolyEvaluation,
    TopologyState,
)

__all__ = [
    "BoundaryStatement",
    "ContextJudgment",
    "Decision",
    "Dimension",
    "GeometricArrangement",
    "GlobalValue",
    "MetaConfig",
    "MetaCycle",
    "MetaResult",
    "Observation",
    "PolyEvaluation",
    "TopologyState",
]

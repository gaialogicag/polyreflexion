"""Shared factory for wiring the polycontextural meta controller.

Centralises controller construction so ``ask_meta.py``, ``run_meta_reflexion.py``,
and the OpenToM benchmark runner all use the same judge-feedback boundaries,
interpretation rules, and strategy planner — while still allowing optional
prompt overrides (e.g. answer-oriented templates in ``ask_meta`` only).
"""

from __future__ import annotations

from collections.abc import Callable

from polyrx.engine import ReflexionEngine
from polyrx.meta.controller import MetaController
from polyrx.meta.datatypes import MetaConfig
from polyrx.meta.judges import PolyJudge
from polyrx.meta.prompts import MetaPromptRegistry

EngineFactory = Callable[[int], ReflexionEngine]


def build_meta_controller(
    engine_factory: EngineFactory,
    judge_client,
    *,
    boundary_client=None,
    config: MetaConfig | None = None,
    meta_prompts: MetaPromptRegistry,
) -> MetaController:
    """Create a ``MetaController`` with the standard meta-layer stack."""
    boundary = boundary_client if boundary_client is not None else judge_client
    return MetaController(
        engine_factory,
        PolyJudge(judge_client, prompts=meta_prompts),
        boundary_client=boundary,
        config=config or MetaConfig(),
        prompts=meta_prompts,
    )

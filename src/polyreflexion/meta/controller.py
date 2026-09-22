"""The meta-controller: observe -> evaluate -> reorganize, in a bounded loop.

The controller never edits reasoning text.  Per cycle it:

1. runs the engine on the current input(s) (expansion may run two in parallel),
2. has every candidate judged from the three contexts and *selects* the best
   (most positive verdicts, ties -> latest),
3. interprets the triple into a global value (with the (A,W,W) rule),
4. records the geometry move, and
5. asks the strategy planner for the next action.

Guards: hard cycle budget, oscillation detection on (value, summary hash),
semantic drift checks, a single F-reset, a low-confidence W rule, and an
engine-failure catch that finishes with the best previous cycle.
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol

from polyreflexion.engine import ReflexionResult
from polyreflexion.meta.boundaries import BoundaryGenerator
from polyreflexion.meta.datatypes import (
    GlobalValue,
    MetaConfig,
    MetaCycle,
    MetaResult,
    Observation,
    summary_hash,
)
from polyreflexion.meta.geometry import GeometryMapper
from polyreflexion.meta.interpretation import Interpreter
from polyreflexion.meta.judges import PolyJudge
from polyreflexion.meta.prompts import MetaPromptRegistry


class ReasoningPrimitive(Protocol):
    """Minimal contract the meta layer needs from any reasoning engine."""

    def run(self, text: str) -> ReflexionResult: ...


# Factory receives the desired reflexion depth and returns a fresh engine.
EngineFactory = Callable[[int], ReasoningPrimitive]

_LOW_CONFIDENCE_RATIONALE = (
    "Low-confidence full-positive triple (>=2 judge fallbacks): dialectical "
    "verdict withdrawn; convergence must not rest on broken judges."
)


class MetaController:
    """Orchestrate the polycontextural meta loop around a black-box engine."""

    def __init__(
        self,
        engine_factory: EngineFactory,
        judge: PolyJudge,
        boundary_client,  # LLMClient-compatible; also used for drift checks
        *,
        config: MetaConfig | None = None,
        prompts: MetaPromptRegistry | None = None,
    ) -> None:
        # Local import avoids a hard module cycle if strategy ever needs the controller.
        from polyreflexion.meta.strategy import StrategyPlanner

        self._engine_factory = engine_factory
        self._judge = judge
        self._config = config or MetaConfig()
        self._prompts = prompts or MetaPromptRegistry()
        self._interpreter = Interpreter(
            reinterpret_full_positive_on_cycle0=(
                self._config.reinterpret_full_positive_on_cycle0
            )
        )
        self._geometry = GeometryMapper()
        self._planner = StrategyPlanner(
            self._prompts,
            BoundaryGenerator(boundary_client, self._prompts),
            self._config,
        )
        self._drift_client = boundary_client

    # ------------------------------------------------------------------ loop

    def run(self, text: str) -> MetaResult:
        cycles: list[MetaCycle] = []
        inputs: list[str] = [text]
        depth = self._config.engine_depth
        previous_summary: str | None = None
        seen_states: set[tuple[GlobalValue, str]] = set()
        reset_used = False
        drift_streak = 0
        termination = "budget_exhausted"
        last_good: MetaCycle | None = None  # last non-drifted cycle

        for index in range(self._config.max_cycles):
            # 1. Run the engine — a crash must never take down the meta loop.
            try:
                candidates = self._run_engine_batch(inputs, depth)
            except Exception as exc:
                termination = f"engine_failure: {exc}"
                break

            # 2. Judge every candidate; select, never edit.
            judged: list[tuple[str, ReflexionResult, object]] = []
            for candidate_input, result in candidates:
                observation = Observation(
                    input_text=text,
                    summary=result.summary,
                    corner_answers=result.corner_answers,
                    previous_summary=previous_summary,
                )
                judged.append((candidate_input, result, self._judge.evaluate(observation)))
            best = max(
                range(len(judged)),
                key=lambda i: (len(judged[i][2].positives()), i),  # ties -> latest
            )
            chosen_input, result, evaluation = judged[best]

            # 3. Interpret the triple (includes the (A,W,W) -> (A,W,R) rule).
            global_value, evaluation = self._interpreter.interpret(evaluation, index)

            # Low-confidence guard: a W carried by >=2 fallback judges must not
            # terminate the loop; it is downgraded to R (continue exploring).
            if global_value is GlobalValue.W and evaluation.fallback_count() >= 2:
                evaluation = evaluation.with_negative_dialectical(
                    _LOW_CONFIDENCE_RATIONALE, method="fallback"
                )
                global_value = GlobalValue.R

            # 4. Geometry move (path continues from the previous cycle).
            previous_path = cycles[-1].topology.path if cycles else ()
            topology = self._geometry.map(
                global_value, frozenset(evaluation.positives()), previous_path
            )

            # Drift check: only meaningful once we left the original input.
            drifted = (
                self._config.drift_check_enabled
                and index > 0
                and not self._on_topic(text, result.summary)
            )

            cycle = MetaCycle(
                index=index,
                input_text=chosen_input,
                all_inputs=tuple(inputs),
                result=result,
                evaluation=evaluation,
                global_value=global_value,
                topology=topology,
                drifted=drifted,
                candidate_summaries=[r.summary for _, r, _ in judged],
            )
            cycles.append(cycle)

            if drifted:
                drift_streak += 1
                if drift_streak >= self._config.max_consecutive_drifts:
                    cycle.action = "terminate"
                    termination = "semantic_drift"
                    break
            else:
                drift_streak = 0
                last_good = cycle

            # Convergence: (A,W,W) on meta cycle 1+ stops immediately.
            # Cycle 0 (A,W,W) is reinterpreted to (A,W,R) above and never reaches
            # this branch with GlobalValue W.
            if global_value is GlobalValue.W and index > 0 and not drifted:
                cycle.action = "terminate"
                termination = "converged_W"
                break

            # Oscillation: identical (value, answer) state seen before.
            state_key = (global_value, summary_hash(result.summary))
            if state_key in seen_states:
                cycle.action = "terminate"
                termination = "oscillation"
                break
            seen_states.add(state_key)

            # 5. Decide the next move — drifted answers are never built upon:
            # planning falls back to the last on-topic cycle instead.
            basis = last_good if last_good is not None else cycle
            decision = self._planner.decide(
                global_value=basis.global_value,
                evaluation=basis.evaluation,
                question=text,
                answer=basis.result.summary,
                reset_used=reset_used,
            )
            cycle.action = decision.action
            cycle.boundaries = decision.boundaries
            cycle.next_inputs = list(decision.next_inputs)

            if decision.action == "terminate":
                termination = decision.reason or "terminated"
                break
            if decision.action == "reset":
                reset_used = True

            inputs = decision.next_inputs
            depth = decision.engine_depth
            previous_summary = result.summary

        return self._finalize(text, cycles, termination)

    def continue_from_summary(
        self,
        text: str,
        seed_summary: str,
        *,
        additional_cycles: int = 1,
        start_index: int = 0,
    ) -> MetaResult:
        """Continue the meta loop from a prior budget's final summary.

        Judges the seed once, plans the next action, then runs
        ``additional_cycles`` more observe→evaluate→decide iterations.
        The seed is kept as cycle ``start_index`` so final selection can
        still prefer it when the continuation does not improve.
        """
        if additional_cycles < 1:
            raise ValueError("additional_cycles must be >= 1")

        from polyreflexion.engine import Node, Perspective, ReflexionResult, Region

        corners = {p: "(seed — no corner artifacts)" for p in Perspective}
        seed_result = ReflexionResult(
            tree=Node(region=Region.root(), depth=0, perspectives=corners),
            corner_answers=corners,
            summary=seed_summary,
        )
        observation = Observation(
            input_text=text,
            summary=seed_summary,
            corner_answers=corners,
            previous_summary=None,
        )
        evaluation = self._judge.evaluate(observation)
        # Interpret as a late cycle so (A,W,W) is true convergence, not cycle-0 reframe.
        global_value, evaluation = self._interpreter.interpret(evaluation, start_index)
        if global_value is GlobalValue.W and evaluation.fallback_count() >= 2:
            evaluation = evaluation.with_negative_dialectical(
                _LOW_CONFIDENCE_RATIONALE, method="fallback"
            )
            global_value = GlobalValue.R

        topology = self._geometry.map(
            global_value, frozenset(evaluation.positives()), ()
        )
        seed_cycle = MetaCycle(
            index=start_index,
            input_text=text,
            all_inputs=(text,),
            result=seed_result,
            evaluation=evaluation,
            global_value=global_value,
            topology=topology,
            drifted=False,
            candidate_summaries=[seed_summary],
            action="seed_continue",
        )

        # Always attempt further work for the requested budget, even if the seed
        # already looks fully positive (W): treat W as R for planning so the
        # extra cycle can still expand/refine.
        plan_value = GlobalValue.R if global_value is GlobalValue.W else global_value
        decision = self._planner.decide(
            global_value=plan_value,
            evaluation=evaluation,
            question=text,
            answer=seed_summary,
            reset_used=False,
        )
        seed_cycle.action = f"seed_continue→{decision.action}"
        seed_cycle.boundaries = decision.boundaries
        seed_cycle.next_inputs = list(decision.next_inputs)

        if decision.action == "terminate" or not decision.next_inputs:
            # Planner refused to continue; fall back to a single expand-style
            # re-run of the original question with the seed as dialectical context.
            inputs = [text]
            depth = self._config.engine_depth
        else:
            inputs = decision.next_inputs
            depth = decision.engine_depth

        cycles: list[MetaCycle] = [seed_cycle]
        previous_summary: str | None = seed_summary
        seen_states: set[tuple[GlobalValue, str]] = {
            (global_value, summary_hash(seed_summary))
        }
        reset_used = False
        drift_streak = 0
        termination = "budget_exhausted"
        last_good: MetaCycle | None = seed_cycle

        for offset in range(additional_cycles):
            index = start_index + 1 + offset
            try:
                candidates = self._run_engine_batch(inputs, depth)
            except Exception as exc:
                termination = f"engine_failure: {exc}"
                break

            judged: list[tuple[str, ReflexionResult, object]] = []
            for candidate_input, result in candidates:
                observation = Observation(
                    input_text=text,
                    summary=result.summary,
                    corner_answers=result.corner_answers,
                    previous_summary=previous_summary,
                )
                judged.append((candidate_input, result, self._judge.evaluate(observation)))
            best = max(
                range(len(judged)),
                key=lambda i: (len(judged[i][2].positives()), i),
            )
            chosen_input, result, evaluation = judged[best]
            global_value, evaluation = self._interpreter.interpret(evaluation, index)
            if global_value is GlobalValue.W and evaluation.fallback_count() >= 2:
                evaluation = evaluation.with_negative_dialectical(
                    _LOW_CONFIDENCE_RATIONALE, method="fallback"
                )
                global_value = GlobalValue.R

            previous_path = cycles[-1].topology.path if cycles else ()
            topology = self._geometry.map(
                global_value, frozenset(evaluation.positives()), previous_path
            )
            drifted = (
                self._config.drift_check_enabled
                and not self._on_topic(text, result.summary)
            )
            cycle = MetaCycle(
                index=index,
                input_text=chosen_input,
                all_inputs=tuple(inputs),
                result=result,
                evaluation=evaluation,
                global_value=global_value,
                topology=topology,
                drifted=drifted,
                candidate_summaries=[r.summary for _, r, _ in judged],
            )
            cycles.append(cycle)

            if drifted:
                drift_streak += 1
                if drift_streak >= self._config.max_consecutive_drifts:
                    cycle.action = "terminate"
                    termination = "semantic_drift"
                    break
            else:
                drift_streak = 0
                last_good = cycle

            if global_value is GlobalValue.W and not drifted:
                cycle.action = "terminate"
                termination = "converged_W"
                break

            state_key = (global_value, summary_hash(result.summary))
            if state_key in seen_states:
                cycle.action = "terminate"
                termination = "oscillation"
                break
            seen_states.add(state_key)

            basis = last_good if last_good is not None else cycle
            decision = self._planner.decide(
                global_value=basis.global_value,
                evaluation=basis.evaluation,
                question=text,
                answer=basis.result.summary,
                reset_used=reset_used,
            )
            cycle.action = decision.action
            cycle.boundaries = decision.boundaries
            cycle.next_inputs = list(decision.next_inputs)

            if decision.action == "terminate":
                termination = decision.reason or "terminated"
                break
            if decision.action == "reset":
                reset_used = True

            inputs = decision.next_inputs
            depth = decision.engine_depth
            previous_summary = result.summary

        return self._finalize(text, cycles, termination)

    # --------------------------------------------------------------- helpers

    def _run_engine_batch(
        self, inputs: list[str], depth: int
    ) -> list[tuple[str, ReflexionResult]]:
        """Run the engine on every input; expansion inputs run concurrently."""
        if len(inputs) == 1:
            return [self._run_one(inputs[0], depth)]
        with ThreadPoolExecutor(
            max_workers=len(inputs), thread_name_prefix="meta-engine"
        ) as pool:
            futures = [pool.submit(self._run_one, text, depth) for text in inputs]
            return [future.result() for future in futures]

    def _run_one(self, text: str, depth: int) -> tuple[str, ReflexionResult]:
        engine = self._engine_factory(depth)
        try:
            return text, engine.run(text)
        finally:
            close = getattr(engine, "close", None)
            if callable(close):
                close()

    def _on_topic(self, question: str, answer: str) -> bool:
        """Cheap yes/no drift check; fails open so a broken check never kills a run."""
        prompt = self._prompts.drift_check(question=question, answer=answer)
        try:
            reply = self._drift_client.complete(prompt).strip().lower()
        except Exception:
            return True
        return not reply.startswith("no")

    def _finalize(
        self, text: str, cycles: list[MetaCycle], termination: str
    ) -> MetaResult:
        """Select the final answer: most positive judgments, ties -> latest cycle."""
        if not cycles:
            return MetaResult(
                question=text,
                final_answer="",
                selected_cycle=-1,
                termination_reason=termination,
                cycles=[],
            )
        eligible = [c for c in cycles if not c.drifted] or cycles
        selected = max(eligible, key=lambda c: (c.positives_count(), c.index))
        return MetaResult(
            question=text,
            final_answer=selected.result.summary,
            selected_cycle=selected.index,
            termination_reason=termination,
            cycles=cycles,
        )

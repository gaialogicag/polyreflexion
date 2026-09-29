from __future__ import annotations

import contextvars
import json
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, Future
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, TypeVar

__all__ = [
    "UNATTRIBUTED",
    "CallCost",
    "ConditionUsage",
    "Pricer",
    "UsageRecorder",
    "active_recorder",
    "attributed_to",
    "build_pricer",
    "record_call",
    "set_active_recorder",
    "submit_in_context",
]

_R = TypeVar("_R")

#: Where a call lands when nothing said which condition it belongs to.
UNATTRIBUTED = "_unattributed"

#: Our provider names -> the ids ``genai-prices`` knows them by.
_PROVIDER_IDS = {"openai": "openai", "gemini": "google"}

#: The condition the current thread is working on. A worker thread starts with
#: an empty context, so this is set inside the worker, not around the pool.
_condition: contextvars.ContextVar[str] = contextvars.ContextVar(
    "polyrx_condition", default=UNATTRIBUTED
)


def submit_in_context(pool: Executor, fn: Callable[..., _R], *args: object) -> Future[_R]:
    """Submit to a pool, carrying the caller's context into the worker.

    A thread from a pool starts with an empty context, so an attribution set by
    the caller is invisible inside the worker and its calls land under
    ``UNATTRIBUTED``. The engine and the meta layer fan out over their own
    pools, which is most of the work in a run, so without this the cost table
    attributes almost nothing.

    The context is copied per submission: a ``Context`` cannot be entered twice.
    """
    return pool.submit(contextvars.copy_context().run, fn, *args)


@contextmanager
def attributed_to(condition: str) -> Iterator[None]:
    """Attribute every model call made in this block to ``condition``."""
    token = _condition.set(condition)
    try:
        yield
    finally:
        _condition.reset(token)


# ---------------------------------------------------------------------------
# Pricing one call
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallCost:
    """What one call cost, and who said so."""

    amount: float
    #: What produced it, recorded so a number can be traced back. Today always
    #: ``genai-prices``; a second source would name itself here.
    source: str


class Pricer(Protocol):
    """Turns one call's token counts into money, or ``None`` if it cannot."""

    def __call__(
        self,
        *,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
    ) -> CallCost | None: ...


class GenAIPricer:
    """The ``genai-prices`` table, which is maintained and knows about tiers.

    Imported lazily and treated as optional: without it a run still records
    every token, and money falls back to whatever the config states.
    """

    def __init__(self) -> None:
        from genai_prices import Usage, calc_price

        self._usage = Usage
        self._calc = calc_price

    def __call__(
        self,
        *,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int,
        reasoning_tokens: int,
    ) -> CallCost | None:
        provider_id = _PROVIDER_IDS.get(provider)
        if provider_id is None:
            # A locally served model has no price to look up.
            return None
        try:
            calculated = self._calc(
                self._usage(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    cache_read_tokens=cached_tokens or None,
                ),
                model_ref=model,
                provider_id=provider_id,
            )
        except Exception:
            # An unknown model is an ordinary outcome, not a failure: the run
            # reports it as unpriced rather than stopping.
            return None
        return CallCost(float(calculated.total_price), "genai-prices")


def build_pricer() -> Pricer | None:
    """The maintained price table, or ``None`` when it is not installed.

    Without it a run still records every token; only the money is missing, and
    the report says so rather than showing a zero.
    """
    try:
        return GenAIPricer()
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


@dataclass
class ConditionUsage:
    """What one condition spent on one model."""

    model: str = ""
    provider: str = ""
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    #: Part of ``prompt_tokens`` the provider served from its cache, not a
    #: separate total.
    cached_tokens: int = 0
    #: Part of ``completion_tokens`` the model spent thinking rather than
    #: answering. Billed as output; recorded apart so the meta layer's cost can
    #: be split into thinking and answering.
    reasoning_tokens: int = 0
    #: Largest single prompt sent to this model, for checking which price band
    #: a run actually sat in.
    max_prompt_tokens: int = 0
    #: Summed per call. ``None`` when nothing could price this model, which is
    #: not the same as zero: a local model is free, an unknown one is unknown.
    cost: float | None = None
    #: What priced it, or empty when nothing could.
    cost_source: str = ""

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class UsageRecorder:
    """Collects per-condition, per-model usage for one run.

    Every benchmark thread writes here, so the counters are taken under a lock.
    The work is a handful of additions against a call that took a network round
    trip, so the lock is never the bottleneck.
    """

    def __init__(self, pricer: Pricer | None = None, checkpoint_path: Path | None = None) -> None:
        self._lock = threading.Lock()
        self._rows: dict[tuple[str, str], ConditionUsage] = {}
        self._pricer = pricer
        #: Where the running tally is dumped after every call, so a crash
        #: loses at most the one call in flight when it happened, never the
        #: spend already made.
        self._checkpoint_path = checkpoint_path
        self._checkpoint_lock = threading.Lock()
        if checkpoint_path is not None:
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        model: str,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        reasoning_tokens: int = 0,
        condition: str | None = None,
    ) -> None:
        """Add one model call to the tally, priced at its own size."""
        cost = None
        if self._pricer is not None:
            cost = self._pricer(
                model=model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
            )

        key = (condition or _condition.get(), model)
        with self._lock:
            row = self._rows.get(key)
            if row is None:
                row = ConditionUsage(model=model, provider=provider)
                self._rows[key] = row
            row.calls += 1
            row.prompt_tokens += int(prompt_tokens or 0)
            row.completion_tokens += int(completion_tokens or 0)
            row.cached_tokens += int(cached_tokens or 0)
            row.reasoning_tokens += int(reasoning_tokens or 0)
            row.max_prompt_tokens = max(row.max_prompt_tokens, int(prompt_tokens or 0))
            if cost is not None:
                row.cost = (row.cost or 0.0) + cost.amount
                row.cost_source = cost.source

        if self._checkpoint_path is not None:
            self._write_checkpoint()

    def _write_checkpoint(self) -> None:
        """Dump the running tally to disk, replacing the previous checkpoint atomically."""
        assert self._checkpoint_path is not None
        with self._checkpoint_lock:
            by_condition = self.snapshot()
            payload = {
                "updated_at": datetime.now(UTC).isoformat(),
                "totals": asdict(totals(by_condition)),
                "by_condition": by_condition,
            }
            tmp = self._checkpoint_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            tmp.replace(self._checkpoint_path)

    def snapshot(self) -> dict[str, dict]:
        """The tally as ``condition -> {...}``, ready to save.

        A condition whose models could not all be priced keeps a cost of
        ``None`` and names them, so a report says the total is partial rather
        than quietly understating it.
        """
        with self._lock:
            rows = {k: ConditionUsage(**asdict(v)) for k, v in self._rows.items()}

        by_condition: dict[str, dict] = {}
        for (condition, model), row in sorted(rows.items()):
            entry = by_condition.setdefault(
                condition,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "max_prompt_tokens": 0,
                    "cost": 0.0,
                    "cost_sources": [],
                    "unpriced_models": [],
                    "by_model": {},
                },
            )
            entry["calls"] += row.calls
            entry["prompt_tokens"] += row.prompt_tokens
            entry["completion_tokens"] += row.completion_tokens
            entry["cached_tokens"] += row.cached_tokens
            entry["reasoning_tokens"] += row.reasoning_tokens
            entry["max_prompt_tokens"] = max(entry["max_prompt_tokens"], row.max_prompt_tokens)
            if row.cost is None:
                if model not in entry["unpriced_models"]:
                    entry["unpriced_models"].append(model)
            else:
                entry["cost"] += row.cost
                if row.cost_source and row.cost_source not in entry["cost_sources"]:
                    entry["cost_sources"].append(row.cost_source)
            entry["by_model"][model] = asdict(row)

        # A condition whose every model is unpriced has no cost at all, rather
        # than a total of zero that reads like "this method was free".
        for entry in by_condition.values():
            if entry["unpriced_models"] and entry["cost"] == 0.0:
                entry["cost"] = None
        return by_condition


#: Process-wide recorder. The benchmark installs one per run; anything running
#: outside a run still records, into a recorder nobody reads.
_active = UsageRecorder()


def set_active_recorder(recorder: UsageRecorder) -> UsageRecorder:
    """Install the recorder every later :func:`record_call` will write to."""
    global _active
    _active = recorder
    return _active


def active_recorder() -> UsageRecorder:
    """The recorder currently installed."""
    return _active


def record_call(
    *,
    model: str,
    provider: str,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> None:
    """Record one model call against the condition in scope."""
    _active.record(
        model=model,
        provider=provider,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


@dataclass
class UsageTotals:
    """Run-wide totals, for the provenance block."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float | None = None
    #: `genai-prices` quotes in US dollars.
    currency: str = "USD"
    cost_sources: list[str] = field(default_factory=list)
    unpriced_models: list[str] = field(default_factory=list)


def totals(by_condition: dict[str, dict], currency: str = "USD") -> UsageTotals:
    """Sum a tally across every condition."""
    out = UsageTotals(currency=currency)
    priced_any = False
    for entry in by_condition.values():
        out.calls += entry["calls"]
        out.prompt_tokens += entry["prompt_tokens"]
        out.completion_tokens += entry["completion_tokens"]
        out.cached_tokens += entry.get("cached_tokens", 0)
        out.reasoning_tokens += entry.get("reasoning_tokens", 0)
        if entry["cost"] is not None:
            out.cost = (out.cost or 0.0) + entry["cost"]
            priced_any = True
        for source in entry.get("cost_sources", []):
            if source not in out.cost_sources:
                out.cost_sources.append(source)
        for model in entry.get("unpriced_models", []):
            if model not in out.unpriced_models:
                out.unpriced_models.append(model)
    if not priced_any:
        out.cost = None
    return out

"""Run a benchmark across the configured conditions.

Dataset-neutral: items arrive as :class:`~polyrx.datasets.base.Item` from
whichever adapter the dataset config names, and nothing here knows which
benchmark it is running.
"""

from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar, cast

from polyrx.benchmark.judge import Judge, normalize_label
from polyrx.benchmark.meta_trace_io import (
    MetaTraceRecord,
    load_meta_trace,
    meta_trace_path,
    save_meta_trace,
)
from polyrx.conditions import Condition, ConditionRegistry
from polyrx.config import (
    DatasetConfig,
    PathsConfig,
    PostProcessConfig,
    PromptsConfig,
    ReportConfig,
)
from polyrx.datasets.base import DatasetAdapter, Item, get_adapter, sample_by_group
from polyrx.engine import PromptRegistry, ReflexionEngine, StubLLMClient
from polyrx.meta.prompts import MetaPromptRegistry
from polyrx.models.base import LLMClient
from polyrx.models.ollama import extract_final_answer, looks_degenerate
from polyrx.models.registry import get_client
from polyrx.provenance import Provenance
from polyrx.usage import (
    UsageRecorder,
    active_recorder,
    attributed_to,
    build_pricer,
    recover_orphaned_usage,
    set_active_recorder,
)
from polyrx.usage import totals as usage_totals

#: One of the config dataclasses a run file carries.
_ConfigT = TypeVar("_ConfigT")


def _registry(config: BenchmarkConfig | None = None) -> ConditionRegistry:
    """The grid a run resolves names against.

    There is no module-level default: the grid is configuration, loaded from
    ``conf/conditions`` into ``BenchmarkConfig.condition_registry``.
    """
    if config is None or config.condition_registry is None:
        raise ValueError(
            "No condition grid configured. Build one with "
            "polyrx.conditions.registry_from_config(cfg.conditions.conditions) "
            "and pass it as BenchmarkConfig.condition_registry."
        )
    return config.condition_registry


def resolve(condition: str, config: BenchmarkConfig | None = None) -> Condition:
    """Look up one condition by name."""
    return _registry(config).get(condition)


def uses_summary_condition(condition: str, config: BenchmarkConfig) -> bool:
    """Conditions that replace the item text with a pre-built summary."""
    return resolve(condition, config).uses_summary


def summary_key_for_condition(condition: str, config: BenchmarkConfig) -> str:
    """Summary cache key; empty string for direct conditions."""
    return resolve(condition, config).summary_key


@dataclass
class BenchmarkConfig:
    """Configuration for one benchmark run, whatever the dataset."""

    #: Which dataset to score, and how to parse it.
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    num_items: int = 10
    seed: int = 42
    reflexion_depth: int = 1
    max_workers: int = 4
    conditions: tuple[str, ...] = ()
    use_stub: bool = False
    # Isolates summary cache for fresh runs (e.g. "ot_v2"). Empty = legacy paths.
    cache_namespace: str = ""
    # Name of the selected meta prompt set, recorded so a report can say which
    # one produced the numbers. The templates themselves live in ``prompts``.
    meta_prompt_profile: str = "default"
    # Resolved templates, composed by Hydra from conf/prompts/.
    prompts: PromptsConfig = field(default_factory=PromptsConfig)
    # When set, sample ``num_items`` new items excluding these IDs (extend mode).
    # Extend mode: sample new items, skipping these.
    exclude_item_ids: frozenset[str] = frozenset()
    # Merge mode: score exactly these items, so columns stay comparable.
    only_item_ids: frozenset[str] = frozenset()
    # Where results, caches and datasets live. Kept on the config so a run can
    # be redirected (tests, scratch dirs) without changing the process CWD.
    paths: PathsConfig = field(default_factory=PathsConfig)
    # Presentation settings (chart sizes, truncation, progress interval).
    report: ReportConfig = field(default_factory=ReportConfig)
    # Thresholds deciding when a generation is rejected and retried.
    postprocess: PostProcessConfig = field(default_factory=PostProcessConfig)
    # Custom condition grid; ``None`` uses the published one.
    condition_registry: ConditionRegistry | None = None


@dataclass
class BenchmarkRun:
    """Full benchmark output."""

    config: BenchmarkConfig
    results: list[dict]
    summaries: dict[str, dict[str, str]] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    #: Code version, model identifiers, dataset revisions, prompt hashes.
    #: Collected at save time when a runner did not supply one.
    provenance: Provenance | None = None
    #: Condition (or summary key) -> tokens, calls and cost. Summary work is
    #: keyed separately from answering because one summary serves several
    #: conditions, and because a cached one costs nothing the second time.
    usage: dict[str, dict] = field(default_factory=dict)


def _client(alias: str, use_stub: bool) -> LLMClient:
    if use_stub:
        return StubLLMClient()
    return get_client(alias)


def extract_label(prediction: str, label_space: str) -> str:
    """Pull the best-matching allowed label from a free-form model response.

    Prefers the cleaned final answer, then searches the full raw text for the
    *last* occurrence of an allowed label (helps when CoT still contains the label).
    """
    cleaned = extract_final_answer(prediction).strip()
    cleaned = re.sub(r"^Answer:\s*", "", cleaned, flags=re.IGNORECASE).strip()
    allowed = [part.strip() for part in label_space.split(",") if part.strip()]
    if not allowed:
        return cleaned

    allowed_norm = {normalize_label(a): a for a in allowed}
    # Longer labels first so "more full" wins over "full".
    allowed_by_len = sorted(allowed, key=lambda a: len(a), reverse=True)

    lines = [line.strip().strip('"').strip("'") for line in cleaned.splitlines() if line.strip()]
    candidates = [*reversed(lines), cleaned]

    for candidate in candidates:
        key = normalize_label(candidate)
        if key in allowed_norm:
            return allowed_norm[key]

    for candidate in candidates:
        lower = candidate.lower()
        for label in allowed_by_len:
            if normalize_label(label) in lower:
                return label

    # Fall back: last explicit label mention anywhere in the raw response.
    raw_lower = prediction.lower()
    last_pos = -1
    last_label = None
    for label in allowed_by_len:
        pos = raw_lower.rfind(normalize_label(label))
        if pos > last_pos:
            last_pos = pos
            last_label = label
    if last_label is not None:
        return last_label

    return cleaned.splitlines()[-1].strip() if lines else cleaned


def _qa_context(item: Item, context: str) -> str:
    choices = item.metadata.get("choices") if item.metadata else None
    if not choices:
        return context
    rendered = "\n".join(f"({letter}) {text}" for letter, text in choices.items())
    return f"{context}\n\nAnswer choices:\n{rendered}"


def _answer_question(
    client: LLMClient,
    prompts: PromptRegistry,
    *,
    context: str,
    item: Item,
) -> str:
    prompt = prompts.qa(
        context=context,
        question=item.question,
        label_space=item.label_space,
    )
    raw = client.complete(prompt).strip()
    return extract_label(raw, item.label_space)


def _backends_for_conditions(conditions: tuple[str, ...], config: BenchmarkConfig) -> set[str]:
    return {resolve(c, config).backend for c in conditions}


def _summary_cache_path(
    summary_key: str,
    item_id: str,
    depth: int,
    *,
    namespace: str = "",
    paths: PathsConfig | None = None,
) -> Path:
    """Disk cache for one story summary.

    The namespace exists so that changing a prompt cannot silently reuse
    summaries built by the previous version of it — the single easiest way to
    publish a number that no longer corresponds to any code.
    """
    root = (paths or PathsConfig()).resolved("cache_dir") / "summaries"
    folder = f"{summary_key}_d{depth}"
    cache_dir = root / namespace / folder if namespace else root / folder
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{item_id}.txt"


def _print_running_total() -> None:
    """Print the run's live token/cost tally so far."""
    spent = usage_totals(active_recorder().snapshot())
    cost = f"${spent.cost:.4f}" if spent.cost is not None else "unpriced"
    print(
        f"    running total: {spent.calls} calls, "
        f"{spent.prompt_tokens + spent.completion_tokens} tokens, {cost}"
    )


def _write_accuracy_checkpoint(path: Path, tally: dict[str, dict[str, int]]) -> None:
    """Dump each condition's running accuracy so far, atomically.

    Grading isn't cached or resumed the way summaries are -- each run always
    regrades everything itself -- so unlike the usage checkpoint this needs no
    recovery story, just a live view of the same tally every process already
    holds in memory.
    """
    payload = {
        condition: {
            "correct": row["correct"],
            "total": row["total"],
            "accuracy": row["correct"] / row["total"] if row["total"] else None,
        }
        for condition, row in tally.items()
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _run_conditions(
    items: list[Item],
    config: BenchmarkConfig,
    prompts: PromptRegistry,
    clients: dict[str, LLMClient],
    judge: Judge,
    accuracy_path: Path,
) -> tuple[dict[tuple[str, str], dict], dict[str, dict[str, str]]]:
    """Build summaries and grade answers together, one item at a time.

    Direct (no-summary) conditions have no per-item build dependency, so they
    grade immediately and concurrently. A summary-based condition grades right
    after its item's summary is built, rather than after every item across
    every condition has one -- accuracy is visible from the first graded
    question, not only once the whole run is nearly done.
    """
    results_by_key: dict[tuple[str, str], dict] = {}
    summaries: dict[str, dict[str, str]] = defaultdict(dict)
    accuracy_tally: dict[str, dict[str, int]] = {}
    total_tasks = len(items) * len(config.conditions)
    done = 0
    # `grade` now runs inside worker threads for the direct-condition pool
    # below (unlike the old code, where only `as_completed` -- the calling
    # thread -- ever touched shared state), so every mutation of
    # results_by_key/accuracy_tally/done and the checkpoint write itself needs
    # to be serialized, not just the checkpoint's own tmp-file replace.
    state_lock = threading.Lock()

    def grade(item: Item, condition: str, context: str) -> None:
        nonlocal done
        backend = resolve(condition, config).backend
        client = clients[backend]
        with attributed_to(condition):
            prediction = _answer_question(client, prompts, context=context, item=item)
            verdict = judge.evaluate(prediction=prediction, item=item)
        with state_lock:
            key = (item.item_id, item.question)
            row = results_by_key.setdefault(
                key,
                {
                    "item_id": item.item_id,
                    # The unit that was sampled. `item_id` identifies the
                    # question, so without this a saved run cannot say how many
                    # stories it drew -- which is what `num_items` means on a
                    # dataset with `sample_by: context`.
                    "context_id": sampling_group(item, config.dataset.sample_by),
                    "question": item.question,
                    "gold_label": item.gold_label,
                    "group": item.group,
                    "label_space": item.label_space,
                    "predictions": {},
                    "judgments": {},
                },
            )
            row["predictions"][condition] = prediction
            row["judgments"][condition] = verdict
            cond_tally = accuracy_tally.setdefault(condition, {"correct": 0, "total": 0})
            cond_tally["total"] += 1
            if verdict.get("correct"):
                cond_tally["correct"] += 1
            # Written after every graded item, same as the usage checkpoint
            # after every call -- not batched to progress_every, so a crash
            # mid-grading loses at most the one item in flight.
            _write_accuracy_checkpoint(accuracy_path, accuracy_tally)
            done += 1
            if done % config.report.progress_every == 0 or done == total_tasks:
                print(f"  answered {done}/{total_tasks}")
                _print_running_total()

    direct_conditions = [c for c in config.conditions if not uses_summary_condition(c, config)]
    if direct_conditions:
        print(f"Grading {len(direct_conditions)} direct condition(s) for {len(items)} items...")
        with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
            futures = [
                pool.submit(grade, item, condition, item.context)
                for item in items
                for condition in direct_conditions
            ]
            for future in as_completed(futures):
                future.result()  # Surfaces a worker's exception instead of swallowing it.

    # summary_key -> (backend, depth, meta_cycles or None)
    jobs: dict[str, tuple[str, int, int | None]] = {}
    # summary_key -> every condition sharing it (an alias means more than one).
    conditions_by_key: dict[str, list[str]] = defaultdict(list)
    for condition in config.conditions:
        if not uses_summary_condition(condition, config):
            continue
        spec = resolve(condition, config)
        depth = spec.depth or config.reflexion_depth
        meta_cycles = spec.meta_cycles if spec.is_meta else None
        jobs[spec.summary_key] = (spec.backend, depth, meta_cycles)
        conditions_by_key[spec.summary_key].append(condition)

    # summary_key -> the next-smaller meta budget's key, when one is configured.
    prior_key_by_summary = {
        spec.summary_key: spec.prior_summary_key
        for spec in (resolve(c, config) for c in config.conditions)
        if spec.prior_summary_key is not None
    }

    for summary_key, (backend, depth, meta_cycles) in sorted(
        jobs.items(), key=lambda kv: (kv[1][1], kv[0])
    ):
        is_meta = meta_cycles is not None
        budget_note = f", meta_cycles={meta_cycles}" if is_meta else ""
        conditions_here = conditions_by_key[summary_key]
        print(
            f"Building {'meta-reflexion' if is_meta else 'reflexion'} summaries "
            f"key={summary_key} (backend={backend}, depth={depth}{budget_note}) "
            f"for {len(items)} items, grading {conditions_here} as each completes..."
        )
        client = _client(backend, config.use_stub)
        for i, item in enumerate(items, start=1):
            item_id, context = item.item_id, item.context
            cache_path = _summary_cache_path(
                summary_key, item_id, depth, namespace=config.cache_namespace, paths=config.paths
            )
            if cache_path.exists():
                print(f"  [{summary_key}] story {i}/{len(items)}: {item_id} (cached)")
                summary = cache_path.read_text(encoding="utf-8")
            else:
                print(f"  [{summary_key}] story {i}/{len(items)}: {item_id}")
                with attributed_to(summary_key):
                    if is_meta:
                        seed_summary = None
                        prior_key = prior_key_by_summary.get(summary_key)
                        if prior_key is not None:
                            prior_path = _summary_cache_path(
                                prior_key,
                                item_id,
                                depth,
                                namespace=config.cache_namespace,
                                paths=config.paths,
                            )
                            if prior_path.exists():
                                seed_summary = prior_path.read_text(encoding="utf-8")
                                print(
                                    f"  [{summary_key}] story {i}/{len(items)}: "
                                    f"continuing from {prior_key}"
                                )
                        # `is_meta` is exactly `meta_cycles is not None`; restate it
                        # so the type checker can see the budget is a real number here.
                        assert meta_cycles is not None
                        summary, meta_result = _run_meta_summary(
                            client,
                            prompts,
                            context,
                            depth=depth,
                            max_workers=config.max_workers,
                            use_stub=config.use_stub,
                            max_cycles=meta_cycles,
                            meta_prompts=MetaPromptRegistry(config.prompts.meta),
                            seed_summary=seed_summary,
                        )
                        save_meta_trace(meta_result, meta_trace_path(cache_path))
                    else:
                        summary = _run_reflexion_summary(
                            client,
                            prompts,
                            context,
                            depth=depth,
                            max_workers=config.max_workers,
                        )
                        # One retry with a stricter reminder if the model collapses into token salad.
                        if looks_degenerate(summary, config.postprocess.degeneracy):
                            print(
                                f"  [{summary_key}] story {i}/{len(items)}: "
                                f"degenerate summary, retrying..."
                            )
                            stricter = (
                                context + "\n\nReminder: write plain English about the story only. "
                                "No math, codes, segment IDs, or puzzle formatting."
                            )
                            summary = _run_reflexion_summary(
                                client,
                                prompts,
                                stricter,
                                depth=depth,
                                max_workers=config.max_workers,
                            )
                    cache_path.write_text(summary, encoding="utf-8")
                _print_running_total()
            summaries[item_id][summary_key] = summary
            for condition in conditions_here:
                grade(item, condition, _qa_context(item, summary))

    return results_by_key, dict(summaries)


def _run_reflexion_summary(
    client: LLMClient,
    prompts: PromptRegistry,
    context: str,
    *,
    depth: int,
    max_workers: int,
) -> str:
    """Run one reflexion tree and return the integrated summary text."""
    with ReflexionEngine(
        client,
        prompts=prompts,
        max_depth=depth,
        max_workers=max_workers,
    ) as engine:
        return engine.run(context).summary


def _run_meta_summary(
    client: LLMClient,
    prompts: PromptRegistry,
    context: str,
    *,
    depth: int,
    max_workers: int,
    use_stub: bool,
    max_cycles: int,
    meta_prompts: MetaPromptRegistry,
    seed_summary: str | None = None,
):
    """Build one story context via the polycontextural meta layer.

    When ``seed_summary`` is provided (prior meta budget cache), continues with
    one additional cycle on top of that answer instead of re-running the full
    ``max_cycles`` budget from scratch.

    Returns ``(final_summary, meta_result)`` so callers can persist judge traces.
    """
    from polyrx.meta.datatypes import MetaConfig, MetaResult
    from polyrx.meta.factory import build_meta_controller
    from polyrx.meta.judges import StubMetaClient

    judge_client = StubMetaClient() if use_stub else get_client("meta_judge")

    # meta_judge's client normally doubles as the boundary/refine writer too
    # (build_meta_controller defaults boundary_client to judge_client), which
    # is fine for a text-completion model. An AnyJev-backed meta_judge can
    # only read a probability off a typed question -- it has no way to write
    # a boundary statement or a refined summary -- so that role falls back to
    # the judge client, which is still required to be a real text model.
    from polyrx.models.anyjev_client import AnyJevClient

    boundary_client = get_client("judge") if isinstance(judge_client, AnyJevClient) else None

    def engine_factory(engine_depth: int) -> ReflexionEngine:
        return ReflexionEngine(
            client,
            prompts=prompts,
            max_depth=engine_depth,
            max_workers=max_workers,
        )

    # Full budget when starting cold; continuation only needs the delta cycles.
    run_cycles = max_cycles if not seed_summary else 1
    controller = build_meta_controller(
        engine_factory,
        judge_client,
        boundary_client=boundary_client,
        config=MetaConfig(max_cycles=run_cycles, engine_depth=depth),
        meta_prompts=meta_prompts,
    )
    if seed_summary:
        # Prior budget cN used indices 0..N-1; continuation starts at index N-1
        # as the seed cycle, then runs one new cycle at index N.
        start_index = max(0, max_cycles - 2) if max_cycles >= 2 else 0
        result: MetaResult = controller.continue_from_summary(
            context,
            seed_summary,
            additional_cycles=1,
            start_index=start_index,
        )
    else:
        result = controller.run(context)
    return result.final_answer, result


def _meta_job_for_summary_key(
    summary_key: str,
    config: BenchmarkConfig,
) -> tuple[str, int, int] | None:
    """Return (backend, depth, meta_cycles) for a meta summary cache key."""
    for condition in config.conditions:
        spec = resolve(condition, config)
        if spec.summary_key != summary_key:
            continue
        if not spec.is_meta:
            return None
        return (spec.backend, spec.depth or config.reflexion_depth, spec.meta_cycles)
    return None


def load_meta_traces_for_run(
    run: BenchmarkRun,
    summary_key: str,
) -> dict[str, MetaTraceRecord]:
    """Load cached ``.meta.json`` traces for every story in ``run.summaries``."""

    job = _meta_job_for_summary_key(summary_key, run.config)
    if job is None:
        return {}
    _, depth, _ = job
    traces: dict[str, MetaTraceRecord] = {}
    for item_id in run.summaries:
        cache_path = _summary_cache_path(
            summary_key,
            item_id,
            depth,
            namespace=run.config.cache_namespace,
            paths=run.config.paths,
        )
        trace = load_meta_trace(meta_trace_path(cache_path))
        if trace is not None:
            traces[item_id] = trace
    return traces


def backfill_meta_traces(
    items: dict[str, str],
    config: BenchmarkConfig,
    prompts: PromptRegistry,
    *,
    summary_keys: tuple[str, ...] | None = None,
    force: bool = False,
) -> int:
    """Re-run meta summaries to write missing ``.meta.json`` judge traces."""
    if summary_keys is None:
        summary_keys = tuple(
            sorted(
                {
                    resolve(c, config).summary_key
                    for c in config.conditions
                    if resolve(c, config).is_meta
                }
            )
        )
    written = 0
    for summary_key in summary_keys:
        job = _meta_job_for_summary_key(summary_key, config)
        if job is None:
            continue
        backend, depth, meta_cycles = job
        client = _client(backend, config.use_stub)
        print(f"Backfilling meta traces key={summary_key} for {len(items)} items...")
        for i, (item_id, context) in enumerate(items.items(), start=1):
            cache_path = _summary_cache_path(
                summary_key, item_id, depth, namespace=config.cache_namespace, paths=config.paths
            )
            trace_path = meta_trace_path(cache_path)
            if trace_path.exists() and not force:
                continue
            print(f"  [{summary_key}] story {i}/{len(items)}: {item_id}")
            summary, meta_result = _run_meta_summary(
                client,
                prompts,
                context,
                depth=depth,
                max_workers=config.max_workers,
                use_stub=config.use_stub,
                max_cycles=meta_cycles,
                meta_prompts=MetaPromptRegistry(config.prompts.meta),
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                cache_path.write_text(summary, encoding="utf-8")
            save_meta_trace(meta_result, trace_path)
            written += 1
    return written


def load_dataset_items(config: BenchmarkConfig) -> tuple[list[Item], DatasetAdapter]:
    """Fetch the configured dataset and sample the items this run scores.

    Sampling honours ``dataset.sample_by``: a dataset whose items share a
    passage is sampled by passage, so every question about a text is scored
    together and its summary is built once.
    """
    from polyrx.data.fetch import DatasetFetcher

    adapter = get_adapter(config.dataset)
    data_dir = config.paths.resolved("data_dir")
    fetcher = DatasetFetcher(data_dir, verify=config.dataset.verify_checksums)
    path = fetcher.fetch_dataset(config.dataset)
    all_items = adapter.load_items(path)

    items = sample_by_group(
        all_items,
        count=config.num_items,
        seed=config.seed,
        exclude=set(config.exclude_item_ids) or None,
        only=set(config.only_item_ids) or None,
        group_key=config.dataset.sample_by,
    )
    return items, adapter


def usage_checkpoint_path(config: BenchmarkConfig) -> Path:
    try:
        from hydra.core.hydra_config import HydraConfig

        run_dir = Path(HydraConfig.get().runtime.output_dir)
    except ValueError:
        # Not running under a Hydra job (e.g. run_benchmark called directly).
        run_dir = (
            config.paths.resolved("cache_dir") / "usage" / (config.cache_namespace or "default")
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / f"usage_{unique_stamp(run_dir, 'usage')}.json"
    return run_dir / "usage.json"


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    """Execute a benchmark run for whatever dataset is configured."""
    started = datetime.now(UTC).isoformat()
    # One recorder per run, installed before any client is built so that
    # nothing the run does goes uncounted. Each call is priced as it is made,
    # because a price band depends on that call's own prompt size.
    #
    # One checkpoint file per invocation, filed inside Hydra's own run
    # directory for this job rather than a namespace-keyed tree next to it:
    # two runs sharing a cache_namespace (a rerun, a sweep) would otherwise
    # overwrite each other's spend record the way a namespace's summary cache
    # is deliberately shared.
    #
    # A crashed attempt under the same namespace already spent real money
    # building whatever this run now finds cached for free; fold that spend
    # back in before the first call, or it disappears from the report.
    hydra_root = config.paths.resolved("results_dir") / "hydra"
    seed, recovered_from = recover_orphaned_usage(hydra_root, config.cache_namespace)
    if recovered_from:
        print(
            f"Recovered spend from {len(recovered_from)} crashed run(s) under "
            f"cache_namespace={config.cache_namespace!r}: {[str(p) for p in recovered_from]}"
        )
    recorder = set_active_recorder(
        UsageRecorder(
            build_pricer(),
            checkpoint_path=usage_checkpoint_path(config),
            cache_namespace=config.cache_namespace,
            seed=seed,
        )
    )
    prompts = PromptRegistry(config.prompts.reflexion)
    items, adapter = load_dataset_items(config)

    needed_aliases = _backends_for_conditions(config.conditions, config) | {"judge"}
    clients = {alias: _client(alias, config.use_stub) for alias in sorted(needed_aliases)}
    judge = Judge(clients["judge"], prompts, adapter)

    accuracy_path = usage_checkpoint_path(config).with_name("accuracy.json")
    print(f"Answering {len(items) * len(config.conditions)} item×condition pairs...")
    results_by_key, summaries = _run_conditions(
        items, config, prompts, clients, judge, accuracy_path
    )

    finished = datetime.now(UTC).isoformat()
    return BenchmarkRun(
        config=config,
        results=sorted(results_by_key.values(), key=lambda r: (r["item_id"], r["question"])),
        summaries=summaries,
        started_at=started,
        finished_at=finished,
        usage=recorder.snapshot(),
    )


def _restore_section(node_type: type[_ConfigT], data: object) -> _ConfigT:
    """Rebuild one config dataclass from the plain dict a run file carries.

    `save_run` writes these with `asdict`, which flattens the nested dataclasses
    into dicts. Reading them back by hand means every field added later is
    silently lost, so the structured schema does it instead. A run file written
    by a different version of the schema keeps its defaults rather than
    stopping a report from being produced.
    """
    if not isinstance(data, dict):
        return node_type()
    from omegaconf import OmegaConf

    try:
        restored = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(node_type), data))
        return cast(_ConfigT, restored)
    except Exception as exc:  # pragma: no cover - depends on the file on disk
        print(f"WARNING: {node_type.__name__} in the run file could not be read ({exc}).")
        return node_type()


def sampling_group(item: Item, sample_by: str) -> str:
    """The id of the unit ``sample_by`` draws: a shared context, or the item.

    Mirrors the bucketing in :func:`polyrx.datasets.base.sample_by_group`. Kept
    in step with it deliberately: a run that records a different notion of a
    group from the one it sampled reports a count nobody can reproduce.
    """
    return item.item_id if sample_by == "item" else Item.content_id(item.context)


def sampled_units(results: list[dict]) -> int:
    """How many sampled units a set of result rows covers.

    Falls back to the item id for runs saved before the group was recorded,
    which reads as one unit per row -- wrong, but no more wrong than what those
    files already said.
    """
    return len({row.get("context_id") or row["item_id"] for row in results})


def load_run(path: Path) -> BenchmarkRun:
    """Load a previously saved benchmark JSON run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    config = BenchmarkConfig(
        # These four are saved and were previously dropped on the way back in,
        # so a regenerated report named the default dataset rather than the one
        # that produced the numbers, and a merged run saved that name again.
        dataset=_restore_section(DatasetConfig, cfg.get("dataset")),
        paths=_restore_section(PathsConfig, cfg.get("paths")),
        report=_restore_section(ReportConfig, cfg.get("report")),
        postprocess=_restore_section(PostProcessConfig, cfg.get("postprocess")),
        num_items=cfg.get("num_items", 10),
        seed=cfg.get("seed", 42),
        reflexion_depth=cfg.get("reflexion_depth", 1),
        max_workers=cfg.get("max_workers", 4),
        conditions=tuple(cfg.get("conditions", ())),
        use_stub=cfg.get("use_stub", False),
        cache_namespace=cfg.get("cache_namespace", ""),
        meta_prompt_profile=cfg.get("meta_prompt_profile", "default"),
    )
    return BenchmarkRun(
        config=config,
        results=payload.get("results", []),
        summaries=payload.get("summaries", {}),
        started_at=payload.get("started_at", ""),
        finished_at=payload.get("finished_at", ""),
        usage=payload.get("usage", {}),
        # The report's notes name the models that answered, so a regenerated
        # report needs the run's own provenance rather than a fresh collection
        # describing whatever machine happens to be regenerating it.
        provenance=Provenance.from_dict(payload.get("provenance")),
    )


def merge_runs(base: BenchmarkRun, extra: BenchmarkRun) -> BenchmarkRun:
    """Merge predictions/judgments/summaries from ``extra`` into ``base``."""
    by_key = {(r["item_id"], r["question"]): dict(r) for r in base.results}
    for row in extra.results:
        key = (row["item_id"], row["question"])
        if key not in by_key:
            by_key[key] = {
                "item_id": row["item_id"],
                "context_id": row.get("context_id", ""),
                "question": row["question"],
                "gold_label": row["gold_label"],
                # The grouping dimension is `group` -- it was renamed from
                # `question_type` when the runner stopped being OpenToM-only.
                # This branch only runs for an item the base run never scored,
                # which is exactly what extend mode produces, so the stale name
                # made `extend_from` fail every time it was used.
                "group": row.get("group", ""),
                "label_space": row["label_space"],
                "predictions": {},
                "judgments": {},
            }
        target = by_key[key]
        target.setdefault("predictions", {}).update(row.get("predictions", {}))
        target.setdefault("judgments", {}).update(row.get("judgments", {}))

    merged_summaries: dict[str, dict[str, str]] = defaultdict(dict)
    for source in (base.summaries, extra.summaries):
        for item_id, backends in source.items():
            merged_summaries[item_id].update(backends)

    conditions = tuple(dict.fromkeys([*base.config.conditions, *extra.config.conditions]))
    # Two runs spent two amounts; the merged run spent the sum. Extending adds
    # the cost of the new items to the cost of the old ones.
    merged_usage: dict[str, dict] = {}
    for spent in (base.usage, extra.usage):
        for usage_key, usage_entry in (spent or {}).items():
            target = merged_usage.setdefault(
                usage_key,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "max_prompt_tokens": 0,
                    "cost": None,
                    "cost_sources": [],
                    "unpriced_models": [],
                    "by_model": {},
                },
            )
            for field_name in (
                "calls",
                "prompt_tokens",
                "completion_tokens",
                "cached_tokens",
                "reasoning_tokens",
            ):
                target[field_name] += usage_entry.get(field_name, 0)
            target["max_prompt_tokens"] = max(
                target["max_prompt_tokens"], usage_entry.get("max_prompt_tokens", 0)
            )
            if usage_entry.get("cost") is not None:
                target["cost"] = (target["cost"] or 0.0) + usage_entry["cost"]
            for source in usage_entry.get("cost_sources", []):
                if source not in target["cost_sources"]:
                    target["cost_sources"].append(source)
            for model in usage_entry.get("unpriced_models", []):
                if model not in target["unpriced_models"]:
                    target["unpriced_models"].append(model)
            for model, row in usage_entry.get("by_model", {}).items():
                target["by_model"].setdefault(model, row)

    merged_units = sampled_units(list(by_key.values()))
    return BenchmarkRun(
        config=BenchmarkConfig(
            # Carried over from the run just executed: without these the merged
            # run is saved describing the default dataset, and its report names
            # a dataset it never scored.
            dataset=extra.config.dataset,
            paths=extra.config.paths,
            report=extra.config.report,
            postprocess=extra.config.postprocess,
            # Not saved to disk (see `_RUNTIME_ONLY_FIELDS` below) so `base`
            # never has one, but `extra` is the run that just executed and
            # still holds the live registry -- write_detailed_report needs it
            # in this in-memory merged run, not just in a reloaded one.
            condition_registry=extra.config.condition_registry,
            num_items=merged_units,
            seed=base.config.seed,
            reflexion_depth=max(base.config.reflexion_depth, extra.config.reflexion_depth),
            max_workers=base.config.max_workers,
            conditions=conditions,
            use_stub=base.config.use_stub or extra.config.use_stub,
            cache_namespace=base.config.cache_namespace or extra.config.cache_namespace,
            meta_prompt_profile=extra.config.meta_prompt_profile or base.config.meta_prompt_profile,
        ),
        results=sorted(by_key.values(), key=lambda r: (r["item_id"], r["question"])),
        summaries=dict(merged_summaries),
        usage=merged_usage,
        started_at=base.started_at or extra.started_at,
        finished_at=extra.finished_at or base.finished_at,
    )


#: Config fields that describe *this process* rather than *this experiment*,
#: and so do not belong in a results file.
_RUNTIME_ONLY_FIELDS = (
    "exclude_item_ids",
    "only_item_ids",
    "condition_registry",
    "prompts",
)


def drop_conditions(run: BenchmarkRun, conditions: set[str]) -> BenchmarkRun:
    """Remove predictions, judgments and summaries for the given conditions.

    Used when a run merges into an older one that still carries a condition
    nobody wants reported any more — an expensive depth-3 column, say. Dropping
    it here rather than filtering at report time keeps the saved JSON and the
    report telling the same story.
    """
    if not conditions:
        return run
    # Direct conditions have no summary, so their empty key is filtered out.
    summary_keys = {key for c in conditions if (key := resolve(c, run.config).summary_key)}
    for row in run.results:
        for condition in conditions:
            row.get("predictions", {}).pop(condition, None)
            row.get("judgments", {}).pop(condition, None)
    for backends in run.summaries.values():
        for key in summary_keys:
            backends.pop(key, None)
    kept = tuple(c for c in run.config.conditions if c not in conditions)
    run.config.conditions = kept
    print(f"Dropped conditions: {sorted(conditions)}; kept {kept}")
    return run


def _config_to_dict(config: BenchmarkConfig) -> dict:
    """JSON-safe config dict, without the runtime-only fields."""
    data = asdict(config)
    for name in _RUNTIME_ONLY_FIELDS:
        data.pop(name, None)
    return data


def raw_dir(results_dir: Path, dataset: str) -> Path:
    """Where a dataset's run files live: ``results/raw/<dataset>``."""
    return results_dir / "raw" / dataset


def reports_dir(results_dir: Path, dataset: str) -> Path:
    """Where a dataset's reports live: ``results/reports/<dataset>``."""
    return results_dir / "reports" / dataset


def interactive_dir(results_dir: Path, tool: str) -> Path:
    """Where one interactive tool's traces live: ``results/interactive/<tool>``.

    Indexed by tool rather than by dataset because these runs have no dataset:
    ``polyrx-ask`` and ``polyrx-reflect`` answer a question typed on the command
    line. Filing them under a dataset would assert a provenance they do not
    have.
    """
    return results_dir / "interactive" / tool


def unique_stamp(output_dir: Path, prefix: str) -> str:
    """A run identifier that no existing run in ``output_dir`` already uses.

    Timestamps have second resolution, and a Hydra sweep launches its jobs well
    inside one second — without this, job #2 silently overwrites job #1's
    results file and the sweep looks like it produced one run.
    """
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    if not (output_dir / f"{prefix}_{stamp}.json").exists():
        return stamp
    for suffix in range(1, 1000):
        candidate = f"{stamp}_{suffix}"
        if not (output_dir / f"{prefix}_{candidate}.json").exists():
            return candidate
    raise RuntimeError(f"Could not find a free run name for {prefix}_{stamp} in {output_dir}")


def save_run(run: BenchmarkRun, results_dir: Path) -> tuple[Path, Path]:
    """Persist raw results, and say where the report belongs.

    Both are filed under the dataset that produced them -- raw runs in
    ``results/raw/<dataset>``, reports in ``results/reports/<dataset>`` -- so a
    directory listing answers "what has this dataset been run with" rather than
    mixing every dataset into one folder and leaving the name to do the work.
    The file names keep the dataset prefix too: a run file is often copied or
    attached on its own, and then the folder is not there to say what it is.
    """
    dataset = run.config.dataset.name
    output_dir = raw_dir(results_dir, dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = dataset
    stamp = unique_stamp(output_dir, prefix)
    json_path = output_dir / f"{prefix}_{stamp}.json"
    payload = {
        "config": _config_to_dict(run.config),
        "provenance": (run.provenance or Provenance.collect()).to_dict(),
        "usage": run.usage,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "summaries": run.summaries,
        "results": run.results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    # One directory per run holds every rendering of it: the summary report,
    # the detailed one, and the charts they share. Keying the charts to a run
    # rather than to a report is what stops the two reports each drawing their
    # own byte-identical copy.
    run_home = reports_dir(results_dir, dataset) / stamp
    run_home.mkdir(parents=True, exist_ok=True)
    # The path carries the dataset and the run, so the file does not repeat
    # them: every run directory holds the same two names.
    return json_path, run_home / "report.md"

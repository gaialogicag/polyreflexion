"""Run OpenToM benchmark across direct and reflexion conditions."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polyreflexion.benchmark.judge import OpenToMJudge, normalize_label
from polyreflexion.benchmark.meta_trace_io import (
    MetaTraceRecord,
    load_meta_trace,
    meta_trace_path,
    save_meta_trace,
)
from polyreflexion.benchmark.opentom_loader import (
    OpenToMItem,
    extend_story_sample,
    load_items,
    sample_stories,
)
from polyreflexion.conditions import Condition, ConditionRegistry, default_registry
from polyreflexion.config import PathsConfig
from polyreflexion.engine import PromptRegistry, ReflexionEngine, StubLLMClient
from polyreflexion.models.base import LLMClient
from polyreflexion.models.ollama import extract_final_answer, looks_degenerate
from polyreflexion.models.registry import get_client
from polyreflexion.provenance import Provenance

# The condition grid lives in :mod:`polyreflexion.conditions`; this module only
# resolves names against it.  Everything that used to be a lookup table
# (backend, depth, meta budget, cache key, prior budget) is now a property of
# :class:`~polyreflexion.conditions.Condition`.
CONDITIONS: tuple[str, ...] = default_registry.names()


def _registry(config: BenchmarkConfig | None = None) -> ConditionRegistry:
    """Registry a run should resolve against (custom grids stay possible)."""
    if config is not None and config.condition_registry is not None:
        return config.condition_registry
    return default_registry


def resolve(condition: str, config: BenchmarkConfig | None = None) -> Condition:
    """Look up one condition by name."""
    return _registry(config).get(condition)


def is_reflexion_condition(condition: str) -> bool:
    return resolve(condition).is_reflexion


def uses_summary_condition(condition: str) -> bool:
    """Conditions that replace the item text with a pre-built summary."""
    return resolve(condition).uses_summary


def depth_for_condition(condition: str, default: int = 1) -> int:
    """Reflexion max-depth for a condition (``default`` only for direct runs)."""
    spec = resolve(condition)
    return spec.depth or default


def is_meta_condition(condition: str) -> bool:
    return resolve(condition).is_meta


def meta_cycles_for_condition(condition: str) -> int:
    """Meta cycle budget for a condition; 0 when the meta layer is off."""
    return resolve(condition).meta_cycles


def summary_key_for_condition(condition: str) -> str:
    """Summary cache key; empty string for direct conditions."""
    return resolve(condition).summary_key


def backend_for_condition(condition: str) -> str:
    return resolve(condition).backend


@dataclass
class BenchmarkConfig:
    """Configuration for an OpenToM benchmark run."""

    num_stories: int = 10
    seed: int = 42
    reflexion_depth: int = 1
    max_workers: int = 4
    conditions: tuple[str, ...] = CONDITIONS
    use_stub: bool = False
    # Isolates summary cache for fresh runs (e.g. "ot_v2"). Empty = legacy paths.
    cache_namespace: str = ""
    # Meta prompt profile: default | opentom | gpqa | ask (see meta/profiles.py).
    meta_prompt_profile: str = "default"
    # When set, sample ``num_stories`` new stories excluding these IDs (extend mode).
    exclude_story_ids: frozenset[str] = frozenset()
    # Where results, caches and datasets live. Kept on the config so a run can
    # be redirected (tests, scratch dirs) without changing the process CWD.
    paths: PathsConfig = field(default_factory=PathsConfig)
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


def _answer_question(
    client: LLMClient,
    prompts: PromptRegistry,
    *,
    context: str,
    item: OpenToMItem,
) -> str:
    prompt = prompts.opentom_qa(
        context=context,
        question=item.question,
        label_space=item.label_space,
    )
    raw = client.complete(prompt).strip()
    return extract_label(raw, item.label_space)


def _backends_for_conditions(conditions: tuple[str, ...]) -> set[str]:
    return {resolve(c).backend for c in conditions}


def _summary_cache_path(
    summary_key: str,
    story_id: str,
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
    return cache_dir / f"{story_id}.txt"


def _build_summaries(
    stories: dict[str, str],
    config: BenchmarkConfig,
    prompts: PromptRegistry,
) -> dict[str, dict[str, str]]:
    """Run reflexion (or meta-reflexion) once per story for each summary key."""
    # summary_key -> (backend, depth, meta_cycles or None)
    jobs: dict[str, tuple[str, int, int | None]] = {}
    for condition in config.conditions:
        if not uses_summary_condition(condition):
            continue
        spec = resolve(condition, config)
        depth = spec.depth or config.reflexion_depth
        meta_cycles = spec.meta_cycles if spec.is_meta else None
        jobs[spec.summary_key] = (spec.backend, depth, meta_cycles)

    # summary_key -> the next-smaller meta budget's key, when one is configured.
    prior_key_by_summary = {
        spec.summary_key: spec.prior_summary_key
        for spec in (resolve(c, config) for c in config.conditions)
        if spec.prior_summary_key is not None
    }

    summaries: dict[str, dict[str, str]] = defaultdict(dict)
    for summary_key, (backend, depth, meta_cycles) in sorted(
        jobs.items(), key=lambda kv: (kv[1][1], kv[0])
    ):
        is_meta = meta_cycles is not None
        budget_note = f", meta_cycles={meta_cycles}" if is_meta else ""
        print(
            f"Building {'meta-reflexion' if is_meta else 'reflexion'} summaries "
            f"key={summary_key} (backend={backend}, depth={depth}{budget_note}) "
            f"for {len(stories)} stories..."
        )
        client = _client(backend, config.use_stub)
        for i, (story_id, narrative) in enumerate(stories.items(), start=1):
            cache_path = _summary_cache_path(
                summary_key, story_id, depth, namespace=config.cache_namespace, paths=config.paths
            )
            if cache_path.exists():
                print(f"  [{summary_key}] story {i}/{len(stories)}: {story_id} (cached)")
                summaries[story_id][summary_key] = cache_path.read_text(encoding="utf-8")
                continue
            print(f"  [{summary_key}] story {i}/{len(stories)}: {story_id}")
            if is_meta:
                seed_summary = None
                prior_key = prior_key_by_summary.get(summary_key)
                if prior_key is not None:
                    prior_path = _summary_cache_path(
                        prior_key,
                        story_id,
                        depth,
                        namespace=config.cache_namespace,
                        paths=config.paths,
                    )
                    if prior_path.exists():
                        seed_summary = prior_path.read_text(encoding="utf-8")
                        print(
                            f"  [{summary_key}] story {i}/{len(stories)}: "
                            f"continuing from {prior_key}"
                        )
                summary, meta_result = _run_meta_summary(
                    client,
                    prompts,
                    narrative,
                    depth=depth,
                    max_workers=config.max_workers,
                    use_stub=config.use_stub,
                    max_cycles=meta_cycles,
                    meta_prompt_profile=config.meta_prompt_profile,
                    seed_summary=seed_summary,
                )
                save_meta_trace(meta_result, meta_trace_path(cache_path))
            else:
                summary = _run_reflexion_summary(
                    client,
                    prompts,
                    narrative,
                    depth=depth,
                    max_workers=config.max_workers,
                )
                # One retry with a stricter reminder if the model collapses into token salad.
                if looks_degenerate(summary):
                    print(f"  [{summary_key}] story {i}/{len(stories)}: degenerate summary, retrying...")
                    stricter = (
                        narrative
                        + "\n\nReminder: write plain English about the story only. "
                        "No math, codes, segment IDs, or puzzle formatting."
                    )
                    summary = _run_reflexion_summary(
                        client,
                        prompts,
                        stricter,
                        depth=depth,
                        max_workers=config.max_workers,
                    )
            summaries[story_id][summary_key] = summary
            cache_path.write_text(summary, encoding="utf-8")
    return dict(summaries)


def _run_reflexion_summary(
    client: LLMClient,
    prompts: PromptRegistry,
    narrative: str,
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
        return engine.run(narrative).summary


def _run_meta_summary(
    client: LLMClient,
    prompts: PromptRegistry,
    narrative: str,
    *,
    depth: int,
    max_workers: int,
    use_stub: bool,
    max_cycles: int,
    meta_prompt_profile: str = "default",
    seed_summary: str | None = None,
):
    """Build one story context via the polycontextural meta layer.

    When ``seed_summary`` is provided (prior meta budget cache), continues with
    one additional cycle on top of that answer instead of re-running the full
    ``max_cycles`` budget from scratch.

    Returns ``(final_summary, meta_result)`` so callers can persist judge traces.
    """
    from polyreflexion.meta.datatypes import MetaConfig, MetaResult
    from polyreflexion.meta.factory import build_meta_controller
    from polyreflexion.meta.judges import StubMetaClient
    from polyreflexion.meta.profiles import get_meta_profile

    meta_prompts = get_meta_profile(meta_prompt_profile).build_registry()
    judge_client = StubMetaClient() if use_stub else get_client("meta_judge")

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
        config=MetaConfig(max_cycles=run_cycles, engine_depth=depth),
        meta_prompts=meta_prompts,
    )
    if seed_summary:
        # Prior budget cN used indices 0..N-1; continuation starts at index N-1
        # as the seed cycle, then runs one new cycle at index N.
        start_index = max(0, max_cycles - 2) if max_cycles >= 2 else 0
        result: MetaResult = controller.continue_from_summary(
            narrative,
            seed_summary,
            additional_cycles=1,
            start_index=start_index,
        )
    else:
        result = controller.run(narrative)
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
    for story_id in run.summaries:
        cache_path = _summary_cache_path(
            summary_key,
            story_id,
            depth,
            namespace=run.config.cache_namespace,
            paths=run.config.paths,
        )
        trace = load_meta_trace(meta_trace_path(cache_path))
        if trace is not None:
            traces[story_id] = trace
    return traces


def backfill_meta_traces(
    stories: dict[str, str],
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
        print(f"Backfilling meta traces key={summary_key} for {len(stories)} stories...")
        for i, (story_id, narrative) in enumerate(stories.items(), start=1):
            cache_path = _summary_cache_path(
                summary_key, story_id, depth, namespace=config.cache_namespace, paths=config.paths
            )
            trace_path = meta_trace_path(cache_path)
            if trace_path.exists() and not force:
                continue
            print(f"  [{summary_key}] story {i}/{len(stories)}: {story_id}")
            summary, meta_result = _run_meta_summary(
                client,
                prompts,
                narrative,
                depth=depth,
                max_workers=config.max_workers,
                use_stub=config.use_stub,
                max_cycles=meta_cycles,
                meta_prompt_profile=config.meta_prompt_profile,
            )
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                cache_path.write_text(summary, encoding="utf-8")
            save_meta_trace(meta_result, trace_path)
            written += 1
    return written


def run_benchmark(config: BenchmarkConfig) -> BenchmarkRun:
    """Execute the OpenToM pilot benchmark."""
    started = datetime.now(UTC).isoformat()
    prompts = PromptRegistry()
    all_items = load_items()
    if config.exclude_story_ids:
        items = extend_story_sample(
            all_items,
            set(config.exclude_story_ids),
            num_additional=config.num_stories,
            seed=config.seed,
        )
    else:
        items = sample_stories(all_items, num_stories=config.num_stories, seed=config.seed)

    stories: dict[str, str] = {}
    for item in items:
        stories[item.story_id] = item.narrative

    need_summaries = any(uses_summary_condition(c) for c in config.conditions)
    summaries: dict[str, dict[str, str]] = {}
    if need_summaries:
        summaries = _build_summaries(stories, config, prompts)

    needed_aliases = _backends_for_conditions(config.conditions) | {"judge"}
    clients = {alias: _client(alias, config.use_stub) for alias in sorted(needed_aliases)}
    judge = OpenToMJudge(clients["judge"], prompts)

    tasks = [(item, condition) for item in items for condition in config.conditions]
    results_by_key: dict[tuple[str, str], dict] = {}

    def process(item: OpenToMItem, condition: str) -> tuple[str, str, dict]:
        backend = resolve(condition, config).backend
        client = clients[backend]
        if uses_summary_condition(condition):
            context = summaries[item.story_id][resolve(condition, config).summary_key]
        else:
            context = item.narrative
        prediction = _answer_question(client, prompts, context=context, item=item)
        verdict = judge.evaluate(
            gold=item.gold_label,
            prediction=prediction,
            label_space=item.label_space,
        )
        return item.story_id, item.question, {
            "condition": condition,
            "prediction": prediction,
            "judgment": verdict,
        }

    print(f"Answering {len(tasks)} question×condition pairs...")
    done = 0
    with ThreadPoolExecutor(max_workers=config.max_workers) as pool:
        futures = {
            pool.submit(process, item, condition): (item, condition)
            for item, condition in tasks
        }
        for future in as_completed(futures):
            item, condition = futures[future]
            story_id, question, payload = future.result()
            key = (story_id, question)
            if key not in results_by_key:
                results_by_key[key] = {
                    "story_id": story_id,
                    "question": question,
                    "gold_label": item.gold_label,
                    "question_type": item.question_type,
                    "label_space": item.label_space,
                    "predictions": {},
                    "judgments": {},
                }
            results_by_key[key]["predictions"][condition] = payload["prediction"]
            results_by_key[key]["judgments"][condition] = payload["judgment"]
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  answered {done}/{len(tasks)}")

    finished = datetime.now(UTC).isoformat()
    return BenchmarkRun(
        config=config,
        results=sorted(results_by_key.values(), key=lambda r: (r["story_id"], r["question"])),
        summaries=summaries,
        started_at=started,
        finished_at=finished,
    )


def load_run(path: Path) -> BenchmarkRun:
    """Load a previously saved benchmark JSON run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    config = BenchmarkConfig(
        num_stories=cfg.get("num_stories", 10),
        seed=cfg.get("seed", 42),
        reflexion_depth=cfg.get("reflexion_depth", 1),
        max_workers=cfg.get("max_workers", 4),
        conditions=tuple(cfg.get("conditions", CONDITIONS)),
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
    )


def merge_runs(base: BenchmarkRun, extra: BenchmarkRun) -> BenchmarkRun:
    """Merge predictions/judgments/summaries from ``extra`` into ``base``."""
    by_key = {(r["story_id"], r["question"]): dict(r) for r in base.results}
    for row in extra.results:
        key = (row["story_id"], row["question"])
        if key not in by_key:
            by_key[key] = {
                "story_id": row["story_id"],
                "question": row["question"],
                "gold_label": row["gold_label"],
                "question_type": row["question_type"],
                "label_space": row["label_space"],
                "predictions": {},
                "judgments": {},
            }
        target = by_key[key]
        target.setdefault("predictions", {}).update(row.get("predictions", {}))
        target.setdefault("judgments", {}).update(row.get("judgments", {}))

    merged_summaries: dict[str, dict[str, str]] = defaultdict(dict)
    for source in (base.summaries, extra.summaries):
        for story_id, backends in source.items():
            merged_summaries[story_id].update(backends)

    conditions = tuple(dict.fromkeys([*base.config.conditions, *extra.config.conditions]))
    merged_story_count = len({r["story_id"] for r in by_key.values()})
    return BenchmarkRun(
        config=BenchmarkConfig(
            num_stories=merged_story_count,
            seed=base.config.seed,
            reflexion_depth=max(base.config.reflexion_depth, extra.config.reflexion_depth),
            max_workers=base.config.max_workers,
            conditions=conditions,
            use_stub=base.config.use_stub or extra.config.use_stub,
            cache_namespace=base.config.cache_namespace or extra.config.cache_namespace,
            meta_prompt_profile=extra.config.meta_prompt_profile
            or base.config.meta_prompt_profile,
        ),
        results=sorted(by_key.values(), key=lambda r: (r["story_id"], r["question"])),
        summaries=dict(merged_summaries),
        started_at=base.started_at or extra.started_at,
        finished_at=extra.finished_at or base.finished_at,
    )


#: Config fields that describe *this process* rather than *this experiment*,
#: and so do not belong in a results file.
_RUNTIME_ONLY_FIELDS = ("exclude_story_ids", "condition_registry")


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


def save_run(run: BenchmarkRun, output_dir: Path) -> tuple[Path, Path]:
    """Persist raw JSON results and return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"opentom_{stamp}.json"
    payload = {
        "config": _config_to_dict(run.config),
        "provenance": (run.provenance or Provenance.collect()).to_dict(),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "summaries": run.summaries,
        "results": run.results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path, output_dir / f"opentom_report_{stamp}.md"

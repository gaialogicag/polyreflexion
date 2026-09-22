"""Run GPQA Diamond benchmark across direct, reflexion, and meta conditions."""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from polyreflexion.benchmark.gpqa_judge import GPQAJudge, extract_choice_letter
from polyreflexion.benchmark.gpqa_loader import (
    GPQAItem,
    domain_by_question_id,
    extend_question_sample,
    load_items,
    sample_questions,
)
from polyreflexion.benchmark.runner import (
    CONDITIONS,
    BenchmarkConfig,
    BenchmarkRun,
    _backends_for_conditions,
    _build_summaries,
    _client,
    resolve,
    uses_summary_condition,
)
from polyreflexion.config import PathsConfig
from polyreflexion.engine import PromptRegistry
from polyreflexion.models.base import LLMClient
from polyreflexion.provenance import Provenance
from polyreflexion.resources import prompt_path

#: Engine + question-answering templates tuned for GPQA.
GPQA_PROMPTS_PATH = prompt_path("reflexion_gpqa.json")


@dataclass
class GPQABenchmarkConfig:
    """Configuration for a GPQA Diamond benchmark run."""

    num_questions: int = 10
    seed: int = 42
    reflexion_depth: int = 1
    max_workers: int = 4
    conditions: tuple[str, ...] = CONDITIONS
    use_stub: bool = False
    cache_namespace: str = ""
    # Default GPQA meta profile (scientific reasoning judges / boundaries).
    meta_prompt_profile: str = "gpqa"
    exclude_question_ids: frozenset[str] = frozenset()
    # When set, evaluate exactly these IDs (e.g. reuse items from a merge-from run).
    fixed_question_ids: frozenset[str] = frozenset()
    # Where results, caches and datasets live.
    paths: PathsConfig = field(default_factory=PathsConfig)


@dataclass
class GPQABenchmarkRun:
    """Full GPQA benchmark output (metrics-compatible result rows)."""

    config: GPQABenchmarkConfig
    results: list[dict]
    summaries: dict[str, dict[str, str]] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    #: Code version, model identifiers, dataset revisions, prompt hashes.
    provenance: Provenance | None = None


def _to_shared_config(config: GPQABenchmarkConfig) -> BenchmarkConfig:
    """Map GPQA config onto the shared summary-builder config shape."""
    return BenchmarkConfig(
        num_stories=config.num_questions,
        seed=config.seed,
        reflexion_depth=config.reflexion_depth,
        max_workers=config.max_workers,
        conditions=config.conditions,
        use_stub=config.use_stub,
        cache_namespace=config.cache_namespace,
        meta_prompt_profile=config.meta_prompt_profile,
        paths=config.paths,
    )


def _answer_question(
    client: LLMClient,
    prompts: PromptRegistry,
    *,
    context: str,
    item: GPQAItem,
) -> str:
    prompt = prompts.gpqa_qa(
        context=context,
        question=item.prompt_text,
        label_space=item.label_space,
    )
    raw = client.complete(prompt).strip()
    return extract_choice_letter(raw, item.label_space)


def run_benchmark(config: GPQABenchmarkConfig) -> GPQABenchmarkRun:
    """Execute the GPQA Diamond benchmark."""
    started = datetime.now(UTC).isoformat()
    prompts = PromptRegistry(GPQA_PROMPTS_PATH)
    all_items = load_items(shuffle_seed=config.seed)
    if config.fixed_question_ids:
        by_id = {item.question_id: item for item in all_items}
        missing = sorted(qid for qid in config.fixed_question_ids if qid not in by_id)
        if missing:
            raise ValueError(f"fixed_question_ids not found in dataset: {missing[:5]}")
        items = [by_id[qid] for qid in sorted(config.fixed_question_ids)]
    elif config.exclude_question_ids:
        items = extend_question_sample(
            all_items,
            set(config.exclude_question_ids),
            num_additional=config.num_questions,
            seed=config.seed,
        )
    else:
        items = sample_questions(
            all_items, num_questions=config.num_questions, seed=config.seed
        )

    # One "story" per question: reflexion summarizes the full MCQ prompt.
    problems: dict[str, str] = {item.question_id: item.prompt_text for item in items}

    need_summaries = any(uses_summary_condition(c) for c in config.conditions)
    summaries: dict[str, dict[str, str]] = {}
    if need_summaries:
        summaries = _build_summaries(problems, _to_shared_config(config), prompts)

    needed_aliases = _backends_for_conditions(config.conditions) | {"judge"}
    clients = {alias: _client(alias, config.use_stub) for alias in sorted(needed_aliases)}
    judge = GPQAJudge(clients["judge"], prompts)

    tasks = [(item, condition) for item in items for condition in config.conditions]
    results_by_key: dict[tuple[str, str], dict] = {}

    def process(item: GPQAItem, condition: str) -> tuple[str, str, dict]:
        backend = resolve(condition).backend
        client = clients[backend]
        if uses_summary_condition(condition):
            context = summaries[item.question_id][resolve(condition).summary_key]
        else:
            context = item.prompt_text
        prediction = _answer_question(client, prompts, context=context, item=item)
        verdict = judge.evaluate(
            gold=item.gold_label,
            prediction=prediction,
            label_space=item.label_space,
        )
        return item.question_id, item.question, {
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
            question_id, question, payload = future.result()
            key = (question_id, question)
            if key not in results_by_key:
                results_by_key[key] = {
                    # story_id alias keeps OpenToM qualitative helpers reusable.
                    "story_id": question_id,
                    "question_id": question_id,
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
    return GPQABenchmarkRun(
        config=config,
        results=sorted(
            results_by_key.values(),
            key=lambda r: (r["story_id"], r["question"]),
        ),
        summaries=summaries,
        started_at=started,
        finished_at=finished,
    )


def load_run(path: Path) -> GPQABenchmarkRun:
    """Load a previously saved GPQA benchmark JSON run."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cfg = payload["config"]
    config = GPQABenchmarkConfig(
        num_questions=cfg.get("num_questions", cfg.get("num_stories", 10)),
        seed=cfg.get("seed", 42),
        reflexion_depth=cfg.get("reflexion_depth", 1),
        max_workers=cfg.get("max_workers", 4),
        conditions=tuple(cfg.get("conditions", CONDITIONS)),
        use_stub=cfg.get("use_stub", False),
        cache_namespace=cfg.get("cache_namespace", ""),
        meta_prompt_profile=cfg.get("meta_prompt_profile", "gpqa"),
    )
    return GPQABenchmarkRun(
        config=config,
        results=payload.get("results", []),
        summaries=payload.get("summaries", {}),
        started_at=payload.get("started_at", ""),
        finished_at=payload.get("finished_at", ""),
    )


def merge_runs(base: GPQABenchmarkRun, extra: GPQABenchmarkRun) -> GPQABenchmarkRun:
    """Merge predictions/judgments/summaries from ``extra`` into ``base``."""
    by_key = {(r["story_id"], r["question"]): dict(r) for r in base.results}
    for row in extra.results:
        key = (row["story_id"], row["question"])
        if key not in by_key:
            by_key[key] = {
                "story_id": row["story_id"],
                "question_id": row.get("question_id", row["story_id"]),
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
    merged_count = len({r["story_id"] for r in by_key.values()})
    return GPQABenchmarkRun(
        config=GPQABenchmarkConfig(
            num_questions=merged_count,
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


#: Config fields describing this process rather than this experiment.
_RUNTIME_ONLY_FIELDS = ("exclude_question_ids", "fixed_question_ids")


def _config_to_dict(config: GPQABenchmarkConfig) -> dict:
    """JSON-safe config dict, without the runtime-only fields."""
    data = asdict(config)
    for name in _RUNTIME_ONLY_FIELDS:
        data.pop(name, None)
    return data


def save_run(run: GPQABenchmarkRun, output_dir: Path) -> tuple[Path, Path]:
    """Persist raw JSON results and return (json_path, report_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"gpqa_{stamp}.json"
    payload = {
        "config": _config_to_dict(run.config),
        "provenance": (run.provenance or Provenance.collect()).to_dict(),
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "summaries": run.summaries,
        "results": run.results,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return json_path, output_dir / f"gpqa_report_{stamp}.md"


def as_opentom_run(run: GPQABenchmarkRun) -> BenchmarkRun:
    """Adapter for shared report helpers that expect ``BenchmarkRun``."""
    return BenchmarkRun(
        config=_to_shared_config(run.config),
        results=run.results,
        summaries=run.summaries,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


def enrich_run_domains(run: GPQABenchmarkRun) -> GPQABenchmarkRun:
    """Fill ``question_type`` from High-level domain metadata when missing."""
    domains = domain_by_question_id(shuffle_seed=run.config.seed)
    updated = 0
    for row in run.results:
        qid = row.get("question_id") or row.get("story_id")
        domain = domains.get(qid or "")
        if (domain and domain != "unspecified" and row.get("question_type") != domain) or (domain and row.get("question_type") in (None, "", "unspecified")):
            row["question_type"] = domain
            updated += 1
    if updated:
        print(f"Enriched domains on {updated} result rows")
    return run


def drop_conditions(
    run: GPQABenchmarkRun,
    conditions: set[str],
) -> GPQABenchmarkRun:
    """Remove predictions/judgments/summaries for the given conditions."""
    if not conditions:
        return run
    # Direct conditions have no summary, so their empty key is filtered out.
    summary_keys = {key for c in conditions if (key := resolve(c).summary_key)}
    for row in run.results:
        for cond in conditions:
            row.get("predictions", {}).pop(cond, None)
            row.get("judgments", {}).pop(cond, None)
    if summary_keys:
        for backends in run.summaries.values():
            for key in summary_keys:
                backends.pop(key, None)
    kept = tuple(c for c in run.config.conditions if c not in conditions)
    run.config.conditions = kept
    print(f"Dropped conditions: {sorted(conditions)}; kept {kept}")
    return run


def has_labeled_domains(results: list[dict]) -> bool:
    """True when at least one result has a real High-level domain label."""
    return any(
        (row.get("question_type") or "unspecified") != "unspecified"
        for row in results
    )

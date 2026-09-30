"""Single Hydra entry point for every benchmark.

Replaces four argparse scripts (``run_opentom_benchmark.py``,
``run_gpqa_benchmark.py``, ``run_opentom_meta.py``, ``run_gpqa_meta.py``).  The
two ``*_meta.py`` sweep scripts existed only to loop a meta budget and chain
each run onto the previous one's results; Hydra's multirun does the looping and
``experiment.merge_from=latest`` does the chaining.

Examples
--------
Offline smoke test, no API key needed::

    polyrx-bench experiment=smoke

A GPQA run::

    polyrx-bench experiment=gpqa experiment.num_items=100 \\
        experiment.cache_namespace=gpqa_v1 experiment.detailed=true

Incremental meta budgets, each continuing from the previous one's summaries::

    polyrx-bench -m experiment=gpqa experiment.merge_from=latest \\
        'experiment.conditions=[answerer_meta_c1],[answerer_meta_c2],[answerer_meta_c3]'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from polyrx.conditions import (
    LOCK_FILENAME,
    ConditionRegistry,
    GridLockError,
    registry_from_config,
    require_lock_match,
)
from polyrx.conf_store import register
from polyrx.config import HostedModelConfig, RootConfig
from polyrx.env import load_env
from polyrx.models.registry import set_active_backends
from polyrx.provenance import ModelInfo, Provenance, UsageInfo
from polyrx.resources import find_conf_dir
from polyrx.usage import totals as usage_totals

register()

#: ``merge_from=latest`` resolves to the newest run file of the same dataset.
_LATEST = "latest"


def _resolve_run_path(value: str, results_dir: Path, dataset: str) -> Path:
    """Turn ``merge_from`` / ``extend_from`` into a concrete run file.

    Accepts a literal path or the sentinel ``latest``.  Chaining sweeps by hand
    meant pasting a timestamped filename between commands; ``latest`` is what
    removes that step.
    """
    if value != _LATEST:
        path = Path(value)
        if not path.is_file():
            raise SystemExit(f"Run file not found: {path}")
        return path
    from polyrx.benchmark.runner import raw_dir

    home = raw_dir(results_dir, dataset)
    candidates = sorted(home.glob(f"{dataset}_*.json"))
    if not candidates:
        # Runs saved before results were filed per dataset sat flat in
        # `results/`, so a run from then is still resolvable.
        candidates = sorted(results_dir.glob(f"{dataset}_*.json"))
        candidates = [
            p for p in candidates if "_report_" not in p.name and "_detailed_" not in p.name
        ]
    if not candidates:
        raise SystemExit(
            f"No previous {dataset} run found in {home}. "
            f"Run once without merge_from/extend_from first."
        )
    return candidates[-1]


def _condition_registry(cfg: RootConfig) -> ConditionRegistry:
    """Build the run's grid from config and check it against the lock.

    The lock is what stops a quiet change to a condition's cache key from
    orphaning every summary already on disk. ``conditions.enforce_lock=false``
    skips the check, which is only correct while deliberately re-locking.
    """
    registry = registry_from_config(cfg.conditions.conditions)
    if cfg.conditions.enforce_lock:
        lock = cfg.paths.resolved("data_dir") / LOCK_FILENAME
        try:
            require_lock_match(registry, lock)
        except GridLockError as exc:
            raise SystemExit(str(exc)) from None
    return registry


def _expand_conditions(names: list[str], registry: ConditionRegistry) -> tuple[str, ...]:
    """Expand group shorthands (``answerer``, ``answerer_nod3``, ``all``) to real names."""
    expanded: list[str] = []
    for name in names:
        if name in registry.names():
            expanded.append(name)
            continue
        group = registry.expand_group(name)
        if not group:
            raise SystemExit(
                f"Unknown condition or group {name!r}. "
                f"Known conditions: {', '.join(registry.names())}"
            )
        expanded.extend(group)
    return tuple(dict.fromkeys(expanded))


def _require_credentials(cfg: RootConfig) -> None:
    """Fail before any work when a needed key is missing."""
    if cfg.experiment.use_stub:
        return
    # A role served by a local runtime has no key to require, so what needs one
    # is read off each role's config rather than assumed from its name.
    backends = (
        getattr(cfg.backends, name) for name in ("answerer", "judge", "meta_judge", "open_weights")
    )
    missing = sorted(
        {
            backend.api_key_env
            for backend in backends
            if isinstance(backend, HostedModelConfig) and not os.environ.get(backend.api_key_env)
        }
    )
    if missing:
        raise SystemExit(
            f"Missing credentials: {', '.join(missing)}. "
            f"Export them, put them in a .env file, or run with experiment.use_stub=true."
        )


def _prompt_set_name() -> str:
    """Which meta prompt set Hydra selected, for the run record.

    The templates themselves travel in the config; this is only the label a
    report prints, so a reader can say "gpqa set" without diffing the text.
    """
    from hydra.core.hydra_config import HydraConfig

    try:
        return str(HydraConfig.get().runtime.choices["prompts/meta"])
    except Exception:
        return "unknown"


def _model_info(cfg: RootConfig) -> list[ModelInfo]:
    """Record the models that will actually be called, for the run's provenance."""
    infos: list[ModelInfo] = []
    for role in ("answerer", "judge", "meta_judge", "open_weights"):
        backend = getattr(cfg.backends, role)
        # The provider is whatever the config for this role turned out to be --
        # reading it off the type is what keeps the record true for a run that
        # mixes providers, rather than labelling everything as one of them.
        infos.append(
            ModelInfo(
                role=role,
                backend=type(backend).__name__.replace("Config", "").lower(),
                model=backend.model,
                base_url=getattr(backend, "base_url", None) or getattr(backend, "host", None),
                temperature=getattr(backend, "temperature", None),
            )
        )
    return infos


def _run_suite(cfg: RootConfig) -> int:
    """Run the configured dataset through the configured conditions.

    One path for every benchmark: which dataset is being scored is a config
    decision, not a branch in the code.
    """
    from polyrx.benchmark.detailed_report import write_detailed_report
    from polyrx.benchmark.report import write_report
    from polyrx.benchmark.runner import (
        BenchmarkConfig,
        drop_conditions,
        load_run,
        merge_runs,
        run_benchmark,
        save_run,
        usage_checkpoint_path,
    )
    from polyrx.usage import discard_checkpoint

    exp = cfg.experiment
    dataset = cfg.dataset.name
    results_dir = cfg.paths.resolved("results_dir")
    registry = _condition_registry(cfg)
    conditions = _expand_conditions(list(exp.conditions), registry)

    base = None
    exclude_ids: frozenset[str] = frozenset()
    only_ids: frozenset[str] = frozenset()
    if exp.extend_from:
        # Extend: add new items, skipping the ones already scored.
        base = load_run(_resolve_run_path(exp.extend_from, results_dir, dataset))
        exclude_ids = frozenset({r["item_id"] for r in base.results})
    elif exp.merge_from:
        # Merge: add conditions to the same items, or the columns are not
        # comparable with the ones already in the run.
        base = load_run(_resolve_run_path(exp.merge_from, results_dir, dataset))
        only_ids = frozenset({r["item_id"] for r in base.results})

    config = BenchmarkConfig(
        dataset=cfg.dataset,
        num_items=len(only_ids) if only_ids else exp.num_items,
        seed=exp.seed,
        max_workers=exp.max_workers,
        conditions=conditions,
        use_stub=exp.use_stub,
        cache_namespace=exp.cache_namespace,
        meta_prompt_profile=_prompt_set_name(),
        prompts=cfg.prompts,
        exclude_item_ids=exclude_ids,
        only_item_ids=only_ids,
        paths=cfg.paths,
        report=cfg.report,
        postprocess=cfg.postprocess,
        condition_registry=registry,
    )
    run = run_benchmark(config)
    if base is not None:
        run = merge_runs(base, run)
    if exp.drop_conditions:
        run = drop_conditions(run, set(exp.drop_conditions))

    # The provenance block has carried an empty UsageInfo since it was written;
    # fill it from what the run actually recorded.
    spent = usage_totals(run.usage)
    run.provenance = Provenance.collect(
        models=_model_info(cfg),
        prompts=cfg.prompts,
        config=cfg.provenance,
        usage=UsageInfo(
            prompt_tokens=spent.prompt_tokens,
            completion_tokens=spent.completion_tokens,
            api_calls=spent.calls,
            cached_tokens=spent.cached_tokens,
            reasoning_tokens=spent.reasoning_tokens,
            estimated_cost=spent.cost,
            currency=spent.currency,
        ),
    )
    json_path, report_path = save_run(run, results_dir)
    # Now that the run's own spend is durably saved in json_path, its live
    # checkpoint is redundant -- and if left behind, indistinguishable from a
    # crashed run's checkpoint to a later invocation's recovery pass.
    discard_checkpoint(usage_checkpoint_path(config))
    write_report(run, report_path)
    print(f"Saved results: {json_path}")
    print(f"Saved report:  {report_path}")
    if exp.detailed:
        detailed = report_path.with_name("detailed.md")
        write_detailed_report(run, detailed)
        print(f"Saved detailed: {detailed}")
    _print_warnings(run.provenance)
    return 0


def _print_warnings(provenance: Provenance | None) -> None:
    """Say plainly why a run may not be exactly reproducible."""
    issues = provenance.warnings() if provenance else []
    if not issues:
        return
    print("\nReproducibility notes:")
    for issue in issues:
        print(f"  - {issue}")


@hydra.main(version_base="1.3", config_path=None, config_name="config")
def _main(cfg: DictConfig) -> int:
    """Compose the config, install the backends, run the configured dataset."""
    typed: RootConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    _require_credentials(typed)
    set_active_backends(typed.backends, typed.postprocess)

    print(OmegaConf.to_yaml(cfg))
    return _run_suite(typed)


def main() -> int:
    """Point Hydra at the repository's ``conf/`` tree, then run.

    ``@hydra.main`` resolves a relative ``config_path`` against the file it
    decorates, which would tie the tree to the package directory. The tree
    lives at the repository root instead, so it is added to the search path
    here, once, as an absolute location.
    """
    load_env()
    try:
        conf_dir = find_conf_dir()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    sys.argv.insert(1, f"--config-dir={conf_dir}")
    return _main()


if __name__ == "__main__":
    sys.exit(main())

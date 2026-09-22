"""Single Hydra entry point for both benchmark suites.

Replaces four argparse scripts (``run_opentom_benchmark.py``,
``run_gpqa_benchmark.py``, ``run_opentom_meta.py``, ``run_gpqa_meta.py``).  The
two ``*_meta.py`` sweep scripts existed only to loop a meta budget and chain
each run onto the previous one's results; Hydra's multirun does the looping and
``experiment.merge_from=latest`` does the chaining.

Examples
--------
Offline smoke test, no API key needed::

    polyrx-bench experiment=smoke

The published GPQA suite::

    polyrx-bench experiment=gpqa experiment.num_items=100 \\
        experiment.cache_namespace=gpqa_v1 experiment.detailed=true

Incremental meta budgets, each continuing from the previous one's summaries::

    polyrx-bench -m experiment=gpqa experiment.merge_from=latest \\
        'experiment.conditions=[nano_meta_c1],[nano_meta_c2],[nano_meta_c3]'
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from polyreflexion.conditions import default_registry
from polyreflexion.conf_store import register
from polyreflexion.config import RootConfig
from polyreflexion.models.registry import set_active_backends
from polyreflexion.provenance import ModelInfo, Provenance
from polyreflexion.resources import PROMPTS_DIR, find_conf_dir

register()

#: ``merge_from=latest`` resolves to the newest run file of the same suite.
_LATEST = "latest"


def _resolve_run_path(value: str, results_dir: Path, suite: str) -> Path:
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
    candidates = sorted(results_dir.glob(f"{suite}_*.json"))
    # Exclude derived report files that share the prefix.
    candidates = [p for p in candidates if "_report_" not in p.name and "_detailed_" not in p.name]
    if not candidates:
        raise SystemExit(
            f"No previous {suite} run found in {results_dir}. "
            f"Run once without merge_from/extend_from first."
        )
    return candidates[-1]


def _expand_conditions(names: list[str]) -> tuple[str, ...]:
    """Expand group shorthands (``nano``, ``nano_nod3``, ``all``) to real names."""
    expanded: list[str] = []
    for name in names:
        if name in default_registry.names():
            expanded.append(name)
            continue
        group = default_registry.expand_group(name)
        if not group:
            raise SystemExit(
                f"Unknown condition or group {name!r}. "
                f"Known conditions: {', '.join(default_registry.names())}"
            )
        expanded.extend(group)
    return tuple(dict.fromkeys(expanded))


def _require_credentials(cfg: RootConfig) -> None:
    """Fail before any work when a needed key is missing."""
    if cfg.experiment.use_stub:
        return
    missing = sorted(
        {
            backend.api_key_env
            for backend in (cfg.backends.nano, cfg.backends.judge, cfg.backends.meta_judge)
            if not os.environ.get(backend.api_key_env)
        }
    )
    if missing:
        raise SystemExit(
            f"Missing credentials: {', '.join(missing)}. "
            f"Export them, put them in a .env file, or run with experiment.use_stub=true."
        )


def _model_info(cfg: RootConfig) -> list[ModelInfo]:
    """Record the models that will actually be called, for the run's provenance."""
    infos: list[ModelInfo] = []
    for role in ("nano", "judge", "meta_judge"):
        backend = getattr(cfg.backends, role)
        infos.append(
            ModelInfo(
                role=role,
                backend="openai",
                model=backend.model,
                base_url=backend.base_url,
                temperature=backend.temperature,
            )
        )
    infos.append(
        ModelInfo(
            role="phi",
            backend="ollama",
            model=cfg.backends.phi.model,
            base_url=cfg.backends.phi.host,
            temperature=cfg.backends.phi.temperature,
        )
    )
    return infos


def _run_opentom(cfg: RootConfig) -> int:
    from polyreflexion.benchmark.detailed_report import write_detailed_report
    from polyreflexion.benchmark.report import write_report
    from polyreflexion.benchmark.runner import (
        BenchmarkConfig,
        drop_conditions,
        load_run,
        merge_runs,
        run_benchmark,
        save_run,
    )

    exp = cfg.experiment
    results_dir = cfg.paths.resolved("results_dir")
    conditions = _expand_conditions(list(exp.conditions))

    base = None
    exclude_ids: frozenset[str] = frozenset()
    if exp.extend_from:
        base = load_run(_resolve_run_path(exp.extend_from, results_dir, "opentom"))
        exclude_ids = frozenset({r["story_id"] for r in base.results})
    elif exp.merge_from:
        base = load_run(_resolve_run_path(exp.merge_from, results_dir, "opentom"))

    config = BenchmarkConfig(
        num_stories=exp.num_items,
        seed=exp.seed,
        max_workers=exp.max_workers,
        conditions=conditions,
        use_stub=exp.use_stub,
        cache_namespace=exp.cache_namespace,
        meta_prompt_profile=cfg.meta.profile,
        exclude_story_ids=exclude_ids,
        paths=cfg.paths,
    )
    run = run_benchmark(config)
    if base is not None:
        run = merge_runs(base, run)
    if exp.drop_conditions:
        run = drop_conditions(run, set(exp.drop_conditions))

    run.provenance = Provenance.collect(
        models=_model_info(cfg),
        prompt_files=sorted(PROMPTS_DIR.glob("*.json")),
    )
    json_path, report_path = save_run(run, results_dir)
    write_report(run, report_path)
    print(f"Saved results: {json_path}")
    print(f"Saved report:  {report_path}")
    if exp.detailed:
        detailed = results_dir / report_path.name.replace("opentom_report_", "opentom_detailed_")
        write_detailed_report(run, detailed)
        print(f"Saved detailed: {detailed}")
    _print_warnings(run.provenance)
    return 0


def _run_gpqa(cfg: RootConfig) -> int:
    from polyreflexion.benchmark.gpqa_detailed_report import write_detailed_report
    from polyreflexion.benchmark.gpqa_report import write_report
    from polyreflexion.benchmark.gpqa_runner import (
        GPQABenchmarkConfig,
        drop_conditions,
        enrich_run_domains,
        load_run,
        merge_runs,
        run_benchmark,
        save_run,
    )

    exp = cfg.experiment
    results_dir = cfg.paths.resolved("results_dir")
    conditions = _expand_conditions(list(exp.conditions))

    base = None
    exclude_ids: frozenset[str] = frozenset()
    fixed_ids: frozenset[str] = frozenset()
    if exp.extend_from:
        base = load_run(_resolve_run_path(exp.extend_from, results_dir, "gpqa"))
        exclude_ids = frozenset({r["story_id"] for r in base.results})
    elif exp.merge_from:
        # Merging adds conditions to the *same* questions, so reuse the item set
        # rather than resampling — otherwise the columns are not comparable.
        base = load_run(_resolve_run_path(exp.merge_from, results_dir, "gpqa"))
        fixed_ids = frozenset({r["story_id"] for r in base.results})

    config = GPQABenchmarkConfig(
        num_questions=len(fixed_ids) if fixed_ids else exp.num_items,
        seed=exp.seed,
        max_workers=exp.max_workers,
        conditions=conditions,
        use_stub=exp.use_stub,
        cache_namespace=exp.cache_namespace,
        meta_prompt_profile=cfg.meta.profile,
        exclude_question_ids=exclude_ids,
        fixed_question_ids=fixed_ids,
        paths=cfg.paths,
    )
    run = run_benchmark(config)
    if base is not None:
        run = merge_runs(base, run)
    if exp.drop_conditions:
        run = drop_conditions(run, set(exp.drop_conditions))
    run = enrich_run_domains(run)

    run.provenance = Provenance.collect(
        models=_model_info(cfg),
        prompt_files=sorted(PROMPTS_DIR.glob("*.json")),
    )
    json_path, report_path = save_run(run, results_dir)
    write_report(run, report_path)
    print(f"Saved results: {json_path}")
    print(f"Saved report:  {report_path}")
    if exp.detailed:
        detailed = results_dir / report_path.name.replace("gpqa_report_", "gpqa_detailed_")
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


_SUITES = {"opentom": _run_opentom, "gpqa": _run_gpqa}


@hydra.main(version_base="1.3", config_path=None, config_name="config")
def _main(cfg: DictConfig) -> int:
    """Compose the config, install the backends, run the requested suite."""
    typed: RootConfig = OmegaConf.to_object(cfg)  # type: ignore[assignment]
    _require_credentials(typed)
    set_active_backends(typed.backends)

    runner = _SUITES.get(typed.experiment.suite)
    if runner is None:
        raise SystemExit(f"Unknown suite {typed.experiment.suite!r}. Known: {', '.join(_SUITES)}")
    print(OmegaConf.to_yaml(cfg))
    return runner(typed)


def main() -> int:
    """Point Hydra at the repository's ``conf/`` tree, then run.

    ``@hydra.main`` resolves a relative ``config_path`` against the file it
    decorates, which would tie the tree to the package directory. The tree
    lives at the repository root instead, so it is added to the search path
    here, once, as an absolute location.
    """
    try:
        conf_dir = find_conf_dir()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from None
    sys.argv.insert(1, f"--config-dir={conf_dir}")
    return _main()


if __name__ == "__main__":
    sys.exit(main())

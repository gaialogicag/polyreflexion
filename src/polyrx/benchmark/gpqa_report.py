"""Generate markdown reports with charts for GPQA Diamond benchmark runs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from polyrx.benchmark.gpqa_runner import GPQABenchmarkRun, has_labeled_domains
from polyrx.benchmark.metrics import compute_metrics
from polyrx.charts import pyplot
from polyrx.config import ReportConfig


def _ordered_conditions(metrics: dict, order: tuple[str, ...] = ()) -> list[str]:
    """Condition names in report order: the configured grid first, then the rest.

    ``order`` is the run's own grid, so a report renders columns in the order
    the experiment declared rather than in a hard-coded one.
    """
    known = [c for c in order if c in metrics]
    extras = [c for c in metrics if c not in known]
    return known + extras


def _bar_chart_by_condition(
    metrics: dict,
    chart_path: Path,
    report: ReportConfig | None = None,
    order: tuple[str, ...] = (),
) -> None:
    plt = pyplot()
    if plt is None:
        return
    report = report or ReportConfig()
    conditions = _ordered_conditions(metrics, order)
    acc_scores = [metrics[c].accuracy for c in conditions]

    fig, ax = plt.subplots(figsize=(report.chart_width, report.chart_height))
    ax.bar(range(len(conditions)), acc_scores)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("GPQA Diamond accuracy by condition")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _grouped_chart_by_domain(
    metrics: dict,
    chart_path: Path,
    report: ReportConfig | None = None,
    order: tuple[str, ...] = (),
) -> None:
    plt = pyplot()
    if plt is None:
        return
    report = report or ReportConfig()
    domains = sorted({t for m in metrics.values() for t in m.by_type})
    conditions = _ordered_conditions(metrics, order)
    fig, ax = plt.subplots(figsize=(report.grouped_chart_width, report.grouped_chart_height))
    width = 0.18
    for i, condition in enumerate(conditions):
        scores = [metrics[condition].by_type.get(t, {}).get("accuracy", 0.0) for t in domains]
        offsets = [j + (i - len(conditions) / 2) * width for j in range(len(domains))]
        ax.bar(offsets, scores, width, label=condition)

    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("GPQA Diamond accuracy by domain")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _comparison_chart(
    metrics: dict,
    chart_path: Path,
    report: ReportConfig | None = None,
    order: tuple[str, ...] = (),
) -> None:
    plt = pyplot()
    if plt is None:
        return
    report = report or ReportConfig()
    labels = _ordered_conditions(metrics, order)
    scores = [metrics[c].accuracy for c in labels]
    if not labels:
        return

    fig, ax = plt.subplots(figsize=(report.chart_width, report.chart_height))
    ax.bar(range(len(labels)), scores)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("GPQA condition comparison")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _escape_cell(text: str, max_len: int = 80) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def write_report(
    run: GPQABenchmarkRun,
    report_path: Path,
    report: ReportConfig | None = None,
) -> Path:
    """Write markdown report with charts and per-question table."""
    # Presentation settings travel on the run, so a report regenerated later
    # looks the same as the one the run produced.
    report = report or getattr(run.config, "report", None) or ReportConfig()
    # Column order is the run's own grid, not a constant in this file.
    order = tuple(run.config.conditions)
    metrics = compute_metrics(run.results)
    charts_dir = report_path.parent / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_overall = charts_dir / "gpqa_scores_by_condition.png"
    chart_by_domain = charts_dir / "gpqa_scores_by_domain.png"
    chart_compare = charts_dir / "gpqa_direct_vs_reflexion.png"
    _bar_chart_by_condition(metrics, chart_overall)
    show_domains = has_labeled_domains(run.results)
    if show_domains:
        _grouped_chart_by_domain(metrics, chart_by_domain)
    _comparison_chart(metrics, chart_compare)

    def rel(p: Path) -> str:
        """Chart paths in the report are relative to the report itself."""
        return p.relative_to(report_path.parent).as_posix()

    lines = [
        "# GPQA Diamond Benchmark Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Configuration",
        "",
        f"- Questions: {run.config.num_questions}",
        f"- Reflexion depth: {run.config.reflexion_depth}",
        f"- Seed: {run.config.seed}",
        f"- Stub mode: {run.config.use_stub}",
        f"- Meta prompt profile: `{run.config.meta_prompt_profile}`",
        f"- Cache namespace: `{run.config.cache_namespace or '(legacy — no namespace)'}`",
        f"- Started: {run.started_at}",
        f"- Finished: {run.finished_at}",
        "",
        "## Aggregate scores",
        "",
        "| Condition | Accuracy | Correct | Total |",
        "|-----------|----------|---------|-------|",
    ]

    for name in _ordered_conditions(metrics, order):
        m = metrics[name]
        lines.append(f"| {name} | {m.accuracy:.3f} | {m.correct} | {m.total} |")

    present = {c for row in run.results for c in row.get("predictions", {})}
    pred_cols = [c for c in order if c in present] + sorted(c for c in present if c not in order)
    header_preds = " | ".join(pred_cols)
    header_judges = " | ".join(f"J:{c}" for c in pred_cols)
    sep_preds = " | ".join("---" for _ in pred_cols)
    sep_judges = " | ".join("---" for _ in pred_cols)

    chart_lines = [
        "",
        "## Charts",
        "",
        f"![Scores by condition]({rel(chart_overall)})",
        "",
    ]
    if show_domains:
        chart_lines.extend(
            [
                f"![Scores by domain]({rel(chart_by_domain)})",
                "",
            ]
        )
    chart_lines.extend(
        [
            f"![Condition comparison]({rel(chart_compare)})",
            "",
            "## Per-question results",
            "",
        ]
    )
    if show_domains:
        chart_lines.extend(
            [
                f"| ID | Domain | Question | Gold | {header_preds} | {header_judges} |",
                f"|----|--------|----------|------| {sep_preds} | {sep_judges} |",
            ]
        )
    else:
        chart_lines.extend(
            [
                f"| ID | Question | Gold | {header_preds} | {header_judges} |",
                f"|----|----------|------| {sep_preds} | {sep_judges} |",
            ]
        )
    lines.extend(chart_lines)

    for row in run.results:
        preds = row.get("predictions", {})
        judgments = row.get("judgments", {})

        def verdict(cond: str, judgments: dict = judgments) -> str:
            # Bound as a default so the closure captures this row's judgments
            # rather than whatever the loop variable holds when it is called.
            if cond not in judgments:
                return ""
            return "correct" if judgments[cond].get("correct") else "wrong"

        cells = [_escape_cell(row["story_id"][: report.id_display_chars])]
        if show_domains:
            cells.append(_escape_cell(row.get("question_type", "")))
        cells.extend(
            [
                _escape_cell(row["question"][: report.question_display_chars]),
                _escape_cell(row["gold_label"]),
            ]
        )
        cells.extend(_escape_cell(preds.get(c, "")) for c in pred_cols)
        cells.extend(verdict(c) for c in pred_cols)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- GPQA Diamond is used for evaluation only (not for training).",
            "- Primary scoring: exact A–D letter match (LLM judge only on ambiguous text).",
            "- Reflexion / meta conditions use an integrated scientific reasoning summary "
            "as QA context (summary must not state the final letter).",
            "- `nano_reflexion` / `phi_reflexion` = depth 1; `*_d2` = depth 2; `*_d3` = depth 3.",
            "- `nano_meta_c1` / `nano_meta_c2` / `nano_meta_c3` = depth-1 reflexion steered "
            "by the polycontextural meta layer with meta cycle budgets 1, 2, or 3 "
            "(judges: gpt-4o; separate summary caches per budget).",
            "- Legacy `nano_meta` is an alias for `nano_meta_c2`.",
            "- Engine prompts: `prompts_gpqa.json`; meta profile default: `gpqa`.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

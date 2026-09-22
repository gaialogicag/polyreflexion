"""Markdown report with charts for a benchmark run, whatever the dataset."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from polyrx.benchmark.metrics import compute_metrics
from polyrx.benchmark.runner import BenchmarkRun
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
    ax.set_title("Accuracy by condition")
    fig.tight_layout()
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)


def _grouped_chart_by_type(
    metrics: dict,
    chart_path: Path,
    report: ReportConfig | None = None,
    order: tuple[str, ...] = (),
    group_name: str = "group",
) -> None:
    plt = pyplot()
    if plt is None:
        return
    report = report or ReportConfig()
    types = sorted({t for m in metrics.values() for t in m.by_type})
    conditions = _ordered_conditions(metrics, order)
    fig, ax = plt.subplots(figsize=(report.grouped_chart_width, report.grouped_chart_height))
    width = 0.18
    for i, condition in enumerate(conditions):
        scores = [metrics[condition].by_type.get(t, {}).get("accuracy", 0.0) for t in types]
        offsets = [j + (i - len(conditions) / 2) * width for j in range(len(types))]
        ax.bar(offsets, scores, width, label=condition)

    ax.set_xticks(range(len(types)))
    ax.set_xticklabels(types, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy by {group_name}")
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
    ax.set_title("Condition comparison")
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
    run: BenchmarkRun,
    report_path: Path,
    report: ReportConfig | None = None,
) -> Path:
    """Write markdown report with charts and per-question table."""
    # Presentation settings travel on the run, so a report regenerated later
    # looks the same as the one the run produced.
    report = report or getattr(run.config, "report", None) or ReportConfig()
    # Column order is the run's own grid, not a constant in this file.
    order = tuple(run.config.conditions)
    # What the grouping dimension is called, so headings say "by domain" or
    # "by question type" rather than a generic word.
    group_name = run.config.dataset.group_name or "group"
    metrics = compute_metrics(run.results)
    charts_dir = report_path.parent / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    chart_overall = charts_dir / "scores_by_condition.png"
    chart_by_type = charts_dir / "scores_by_group.png"
    chart_compare = charts_dir / "direct_vs_reflexion.png"
    _bar_chart_by_condition(metrics, chart_overall, report, order)
    _grouped_chart_by_type(metrics, chart_by_type, report, order, group_name)
    _comparison_chart(metrics, chart_compare, report, order)

    def rel(p: Path) -> str:
        """Chart paths in the report are relative to the report itself."""
        return p.relative_to(report_path.parent).as_posix()

    lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Configuration",
        "",
        f"- Dataset: {run.config.dataset.name} (adapter {run.config.dataset.adapter})",
        f"- Items: {run.config.num_items}",
        f"- Reflexion depth: {run.config.reflexion_depth}",
        f"- Seed: {run.config.seed}",
        f"- Stub mode: {run.config.use_stub}",
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

    lines.extend(
        [
            "",
            "## Charts",
            "",
            f"![Scores by condition]({rel(chart_overall)})",
            "",
            f"![Scores by {group_name}]({rel(chart_by_type)})",
            "",
            f"![Condition comparison]({rel(chart_compare)})",
            "",
            "## Per-item results",
            "",
            f"| Story | Question | Gold | {header_preds} | {header_judges} |",
            f"|-------|----------|------| {sep_preds} | {sep_judges} |",
        ]
    )

    for row in run.results:
        preds = row.get("predictions", {})
        judgments = row.get("judgments", {})

        def verdict(cond: str, judgments: dict = judgments) -> str:
            # Bound as a default so the closure captures this row's judgments
            # rather than whatever the loop variable holds when it is called.
            if cond not in judgments:
                return ""
            return "correct" if judgments[cond].get("correct") else "wrong"

        cells = [
            _escape_cell(row["item_id"][: report.id_display_chars]),
            _escape_cell(row["question"][: report.question_display_chars]),
            _escape_cell(row["gold_label"]),
        ]
        cells.extend(_escape_cell(preds.get(c, "")) for c in pred_cols)
        cells.extend(verdict(c) for c in pred_cols)
        lines.append("| " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- The dataset is used for evaluation only, never for training.",
            "- Judge model: gpt-5-mini (fallback string match if judge JSON fails).",
            "- Reflexion conditions use the integrated summary as QA context.",
            "- `nano_reflexion` / `phi_reflexion` = depth 1; `*_d2` = depth 2; `*_d3` = depth 3.",
            "- `nano_meta_c1` / `nano_meta_c2` / `nano_meta_c3` = depth-1 reflexion steered "
            "by the polycontextural meta layer with meta cycle budgets 1, 2, or 3 "
            "(judges: gpt-4o; separate summary caches per budget).",
            "- Legacy `nano_meta` is an alias for `nano_meta_c2`.",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

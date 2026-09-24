"""Markdown report with charts for a benchmark run, whatever the dataset."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from polyrx.benchmark.metrics import compute_metrics
from polyrx.benchmark.runner import BenchmarkRun
from polyrx.charts import pyplot
from polyrx.config import DEFAULT_REPORT_SECTIONS, ReportConfig


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


def charts_dir_for(report_path: Path) -> Path:
    """Directory holding one report's charts, keyed on that report's filename.

    A single shared directory means every run overwrites the previous run's
    figures. Because a report links its charts relatively, the old report does
    not lose them -- it silently renders the newest run's figures under its own
    numbers, which is worse than losing them.
    """
    return report_path.parent / "charts" / report_path.stem


@dataclass
class ReportContext:
    """Everything a section builder is allowed to look at.

    Sections take this and return markdown lines. Keeping the inputs in one
    object is what lets a section be reordered, dropped or added without any
    other section knowing.
    """

    run: BenchmarkRun
    report: ReportConfig
    report_path: Path
    #: Condition order for every table and chart: the run's own grid.
    order: tuple[str, ...]
    #: What the grouping dimension is called, so headings say "by domain" or
    #: "by question type" rather than a generic word.
    group_name: str
    metrics: dict
    charts: dict[str, Path]

    def rel(self, path: Path) -> str:
        """Chart paths in the report are relative to the report itself."""
        return path.relative_to(self.report_path.parent).as_posix()

    @property
    def prediction_columns(self) -> list[str]:
        """Conditions actually present in the results, in grid order."""
        present = {c for row in self.run.results for c in row.get("predictions", {})}
        return [c for c in self.order if c in present] + sorted(
            c for c in present if c not in self.order
        )


def _section_configuration(ctx: ReportContext) -> list[str]:
    config = ctx.run.config
    # `num_items` counts sampled units, which is stories on a dataset sampled
    # by context. Saying so, and giving the row count beside it, is the whole
    # difference between "20 items" meaning 20 stories and 460 questions.
    unit = "item" if config.dataset.sample_by == "item" else config.dataset.sample_by
    rows = len(ctx.run.results)
    return [
        "## Configuration",
        "",
        f"- Dataset: {config.dataset.name} (adapter {config.dataset.adapter})",
        f"- Sampled: {config.num_items} {unit}(s), {rows} scored question(s)",
        f"- Reflexion depth: {config.reflexion_depth}",
        f"- Seed: {config.seed}",
        f"- Stub mode: {config.use_stub}",
        f"- Started: {ctx.run.started_at}",
        f"- Finished: {ctx.run.finished_at}",
        "",
    ]


def _section_aggregate_scores(ctx: ReportContext) -> list[str]:
    lines = [
        "## Aggregate scores",
        "",
        "| Condition | Accuracy | Correct | Total |",
        "|-----------|----------|---------|-------|",
    ]
    for name in _ordered_conditions(ctx.metrics, ctx.order):
        m = ctx.metrics[name]
        lines.append(f"| {name} | {m.accuracy:.3f} | {m.correct} | {m.total} |")
    lines.append("")
    return lines


def _section_charts(ctx: ReportContext) -> list[str]:
    return [
        "## Charts",
        "",
        f"![Scores by condition]({ctx.rel(ctx.charts['overall'])})",
        "",
        f"![Scores by {ctx.group_name}]({ctx.rel(ctx.charts['by_group'])})",
        "",
        f"![Condition comparison]({ctx.rel(ctx.charts['comparison'])})",
        "",
    ]


def _section_per_item(ctx: ReportContext) -> list[str]:
    report = ctx.report
    pred_cols = ctx.prediction_columns
    # The first column is the item id, which identifies the question, not the
    # story. When a dataset groups several questions under one context, the
    # context gets a column of its own rather than being implied by a heading.
    grouped = ctx.run.config.dataset.sample_by != "item"
    has_context = any(row.get("context_id") for row in ctx.run.results)
    show_context = grouped and has_context
    unit = ctx.run.config.dataset.sample_by.capitalize() if show_context else ""

    headers = ["Item"] + ([unit] if show_context else []) + ["Question", "Gold"]
    header = " | ".join(headers + pred_cols + [f"J:{c}" for c in pred_cols])
    separator = " | ".join(["---"] * (len(headers) + 2 * len(pred_cols)))
    lines = ["## Per-item results", "", f"| {header} |", f"| {separator} |"]

    for row in ctx.run.results:
        preds = row.get("predictions", {})
        judgments = row.get("judgments", {})

        def verdict(cond: str, judgments: dict = judgments) -> str:
            # Bound as a default so the closure captures this row's judgments
            # rather than whatever the loop variable holds when it is called.
            if cond not in judgments:
                return ""
            return "correct" if judgments[cond].get("correct") else "wrong"

        cells = [_escape_cell(row["item_id"][: report.id_display_chars])]
        if show_context:
            cells.append(_escape_cell(row.get("context_id", "")[: report.id_display_chars]))
        cells.extend(
            [
                _escape_cell(row["question"][: report.question_display_chars]),
                _escape_cell(row["gold_label"]),
            ]
        )
        cells.extend(_escape_cell(preds.get(c, "")) for c in pred_cols)
        cells.extend(verdict(c) for c in pred_cols)
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def _section_cost(ctx: ReportContext) -> list[str]:
    """What each condition spent, beside what it scored.

    An accuracy gain is only worth something against what it cost to get. The
    tokens are what the run actually consumed; the money is those tokens priced
    by the table in the backends config, which is why a model with no price
    configured is named rather than counted as free.
    """
    usage = ctx.run.usage or {}
    if not usage:
        return [
            "## Cost",
            "",
            "No usage recorded. Runs from before cost tracking, and stub runs, have none.",
            "",
        ]

    # `genai-prices` quotes in US dollars.
    currency = "USD"
    metrics = ctx.metrics
    lines = [
        "## Cost",
        "",
        f"| Condition | Calls | Prompt | of which cached | Completion | of which thinking | "
        f"Cost ({currency}) | Per correct answer |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def row(name: str, entry: dict) -> str:
        cost = entry.get("cost")
        correct = metrics[name].correct if name in metrics else 0
        if cost is None:
            money, per = "not priced", "-"
        else:
            money = f"{cost:.4f}"
            per = f"{cost / correct:.4f}" if correct else "-"
        return (
            f"| {name} | {entry.get('calls', 0)} | {entry.get('prompt_tokens', 0):,} | "
            f"{entry.get('cached_tokens', 0):,} | {entry.get('completion_tokens', 0):,} | "
            f"{entry.get('reasoning_tokens', 0):,} | {money} | {per} |"
        )

    # Conditions first, in grid order, then the shared summary work, which is
    # keyed by cache key rather than by condition because one summary serves
    # several conditions.
    answered = [name for name in _ordered_conditions(metrics, ctx.order) if name in usage]
    for name in answered:
        lines.append(row(name, usage[name]))
    for name in sorted(k for k in usage if k not in answered):
        lines.append(row(name, usage[name]))

    unpriced = sorted({m for e in usage.values() for m in e.get("unpriced_models", [])})
    lines.append("")
    if unpriced:
        lines.append(
            f"No price known for {', '.join(unpriced)}, so any total above is partial. A model "
            f"served locally has no price; a hosted one the table does not carry needs "
            f'`pip install -e ".[cost]"` or a newer `genai-prices`.'
        )
        lines.append("")
    sources = sorted({s for e in usage.values() for s in e.get("cost_sources", [])})
    if sources:
        named = {
            "genai-prices": "the maintained `genai-prices` table, which applies a provider's "
            "prompt-size tiers per call",
        }
        lines.append("Priced by " + "; ".join(named.get(s, s) for s in sources) + ".")
        lines.append("")

    biggest = max((e.get("max_prompt_tokens", 0) for e in usage.values()), default=0)
    if biggest:
        lines.append(
            f"Largest single prompt: {biggest:,} tokens, which is what decides the price "
            f"band a call falls in."
        )
        lines.append("")

    cached = sum(e.get("cached_tokens", 0) for e in usage.values())
    prompt = sum(e.get("prompt_tokens", 0) for e in usage.values())
    if cached:
        priced_cached = any(
            len(p) > 2 and p[2] is not None
            for p in getattr(ctx.run.config, "model_prices", {}).values()
        )
        note = (
            "priced at the cached rate"
            if priced_cached
            else "priced at the full input rate, because no cached rate is configured, so "
            "the cost above is an upper bound"
        )
        lines.append(
            f"{cached:,} of {prompt:,} prompt tokens were served from the provider's cache "
            f"({cached / prompt:.0%}), {note}."
        )
        lines.append("")
    lines.append(
        "Summary rows are shared work: one summary serves every condition with that cache "
        "key, and a summary already cached cost this run nothing. Compare methods on a cold "
        "cache, or the cheaper-looking one may simply have run second."
    )
    lines.append("")
    return lines


def _section_notes(ctx: ReportContext) -> list[str]:
    lines = ["## Notes", ""]
    # Which models answered is a fact about this run, so it is read off the
    # run's provenance rather than written into a note that can go stale.
    provenance = ctx.run.provenance
    for model in getattr(provenance, "models", []) or []:
        lines.append(f"- Role `{model.role}`: {model.model} (backend {model.backend}).")
    lines.extend(f"- {note}" for note in ctx.report.notes)
    lines.append("")
    return lines


#: Section name -> builder. A report is these, in the order `report.sections`
#: names them.
SECTION_BUILDERS: dict[str, Callable[[ReportContext], list[str]]] = {
    "configuration": _section_configuration,
    "aggregate_scores": _section_aggregate_scores,
    "charts": _section_charts,
    "per_item": _section_per_item,
    "cost": _section_cost,
    "notes": _section_notes,
}


def write_report(
    run: BenchmarkRun,
    report_path: Path,
    report: ReportConfig | None = None,
) -> Path:
    """Write the markdown report named by ``report.sections``, and its charts."""
    # Presentation settings travel on the run, so a report regenerated later
    # looks the same as the one the run produced.
    report = report or getattr(run.config, "report", None) or ReportConfig()
    sections = list(report.sections) or list(DEFAULT_REPORT_SECTIONS)
    unknown = [name for name in sections if name not in SECTION_BUILDERS]
    if unknown:
        known = ", ".join(sorted(SECTION_BUILDERS))
        raise ValueError(
            f"Unknown report section(s): {', '.join(unknown)}. Known sections: {known}."
        )

    order = tuple(run.config.conditions)
    group_name = run.config.dataset.group_name or "group"
    metrics = compute_metrics(run.results)

    charts_dir = charts_dir_for(report_path)
    charts_dir.mkdir(parents=True, exist_ok=True)
    charts = {
        "overall": charts_dir / "scores_by_condition.png",
        "by_group": charts_dir / "scores_by_group.png",
        "comparison": charts_dir / "direct_vs_reflexion.png",
    }
    # Drawn whether or not the charts section is selected: the detailed report
    # links the same files, and a run that drops the section from its summary
    # should still leave the figures behind.
    _bar_chart_by_condition(metrics, charts["overall"], report, order)
    _grouped_chart_by_type(metrics, charts["by_group"], report, order, group_name)
    _comparison_chart(metrics, charts["comparison"], report, order)

    ctx = ReportContext(
        run=run,
        report=report,
        report_path=report_path,
        order=order,
        group_name=group_name,
        metrics=metrics,
        charts=charts,
    )

    lines = [
        "# Benchmark Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
    ]
    for name in sections:
        lines.extend(SECTION_BUILDERS[name](ctx))

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

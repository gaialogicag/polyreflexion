"""Detailed markdown analysis for GPQA Diamond benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from polyrx.benchmark.dimension_attribution import (
    build_dimension_attribution,
    format_dimension_table,
    format_transition_table,
    repair_dimension_shares,
)
from polyrx.benchmark.gpqa_qualitative import (
    MARKER_SETS,
    domain_error_analysis,
)
from polyrx.benchmark.gpqa_report import write_report
from polyrx.benchmark.gpqa_runner import (
    GPQABenchmarkRun,
    as_opentom_run,
    has_labeled_domains,
)
from polyrx.benchmark.metrics import compute_metrics
from polyrx.benchmark.qualitative_analysis import (
    META_SUMMARY_KEYS,
    REFLEXION_SUMMARY_KEY,
    FlipRecord,
    FlipSummary,
    _escape_cell,
    collect_flips,
    count_by_type,
    meta_adds_markers_vs_reflexion,
    stories_with_markers,
    summary_length_stats,
)
from polyrx.benchmark.runner import load_meta_traces_for_run, summary_key_for_condition
from polyrx.config import ReportConfig


def _pairwise_vs_direct(results: list[dict], condition: str) -> dict[str, int]:
    wins = losses = ties = both_correct = both_wrong = 0
    for row in results:
        j_direct = row.get("judgments", {}).get("nano_direct")
        j_cond = row.get("judgments", {}).get(condition)
        if not j_direct or not j_cond:
            continue
        d_ok = j_direct.get("correct", False)
        c_ok = j_cond.get("correct", False)
        if d_ok and c_ok:
            both_correct += 1
            ties += 1
        elif not d_ok and not c_ok:
            both_wrong += 1
            ties += 1
        elif c_ok and not d_ok:
            wins += 1
        else:
            losses += 1
    return {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
    }


def _type_table(metrics: dict, conditions: list[str]) -> list[str]:
    types = sorted({t for m in metrics.values() for t in m.by_type})
    lines = [
        "| Domain | " + " | ".join(conditions) + " |",
        "|--------|" + "|".join(["---"] * len(conditions)) + "|",
    ]
    for qtype in types:
        cells = []
        for cond in conditions:
            info = metrics[cond].by_type.get(qtype, {})
            if not info:
                cells.append("—")
            else:
                cells.append(f"{info['accuracy']:.3f} ({info['count']})")
        lines.append(f"| {qtype} | " + " | ".join(cells) + " |")
    return lines


def _flip_type_breakdown(records: list[FlipRecord]) -> list[str]:
    if not records:
        return ["- *(none)*"]
    by_type = count_by_type(records)
    return [f"- `{qtype}`: {count}" for qtype, count in by_type.most_common()]


def _flip_examples_table(records: list[FlipRecord], *, report: ReportConfig) -> list[str]:
    if not records:
        return ["*(none)*", ""]
    lines = [
        "| ID | Domain | Gold | Baseline | Condition | Question |",
        "|----|--------|------|----------|-----------|----------|",
    ]
    for rec in records[: report.max_flip_examples]:
        lines.append(
            f"| {rec.story_id[: report.id_display_chars]} | {rec.question_type} | {rec.gold_label} | "
            f"{rec.baseline_pred} | {rec.condition_pred} | "
            f"{_escape_cell(rec.question, report.flip_question_chars)} |"
        )
    if len(records) > report.max_flip_examples:
        lines.append("")
        lines.append(f"*…and {len(records) - report.max_flip_examples} more.*")
    lines.append("")
    return lines


def _render_flip_section(flip: FlipSummary, *, report: ReportConfig) -> list[str]:
    lines = [
        f"### `{flip.condition}` vs `{flip.baseline}` "
        f"(net {flip.net:+d}: {len(flip.gains)} fixed, {len(flip.losses)} regressed)",
        "",
        "#### Wrong → right",
        "",
        "By domain (Biology / Physics / Chemistry when labeled):",
        *_flip_type_breakdown(flip.gains),
        "",
        *_flip_examples_table(flip.gains, report=report),
        "#### Right → wrong",
        "",
        "By domain (Biology / Physics / Chemistry when labeled):",
        *_flip_type_breakdown(flip.losses),
        "",
        *_flip_examples_table(flip.losses, report=report),
    ]
    return lines


def _render_domain_error_section(
    results: list[dict],
    *,
    baseline: str,
    conditions: list[str],
) -> list[str]:
    lines = [
        "Per-domain error movement vs baseline (all GPQA items in the run).",
        "",
    ]
    for cond in conditions:
        if cond == baseline:
            continue
        stats = domain_error_analysis(results, baseline=baseline, condition=cond)
        lines.append(f"### `{cond}` vs `{baseline}`")
        lines.append("")
        lines.append(
            "| Domain | N | Baseline wrong | Condition wrong | Fixed | Regressed | "
            "Both wrong | Δ error rate |"
        )
        lines.append(
            "|--------|---|----------------|-----------------|-------|-----------|"
            "------------|--------------|"
        )
        for domain, s in stats.items():
            delta = float(s["condition_error_rate"]) - float(s["baseline_error_rate"])
            lines.append(
                f"| {domain} | {s['total']} | {s['baseline_wrong']} | "
                f"{s['condition_wrong']} | {s['fixed']} | {s['regressed']} | "
                f"{s['both_wrong']} | {delta:+.1%} |"
            )
        lines.append("")
    return lines


def _render_summary_length_section(
    summaries: dict[str, dict[str, str]],
    *,
    reflexion_key: str = REFLEXION_SUMMARY_KEY,
) -> list[str]:
    keys = [reflexion_key, *META_SUMMARY_KEYS]
    # Bind the filtered list to a new name: reassigning `stats` would keep its
    # original Optional element type and every later attribute access is then
    # only accidentally safe.
    measured = [summary_length_stats(summaries, key) for key in keys]
    stats = [s for s in measured if s is not None]
    if not stats:
        return ["No question summaries available in this run.", ""]
    base = next((s for s in stats if s.key == reflexion_key), stats[0])
    lines = [
        "Integrated reasoning summaries are the context fed to the QA model "
        "(proxy for reasoning depth).",
        "",
        "| Summary key | Questions | Mean words | Mean chars | vs reflexion (words) |",
        "|-------------|-----------|------------|------------|----------------------|",
    ]
    for item in stats:
        delta = item.mean_words - base.mean_words
        delta_cell = "—" if item.key == base.key else f"{delta:+.1f}"
        lines.append(
            f"| {item.key} | {item.story_count} | {item.mean_words:.1f} | "
            f"{item.mean_chars:.0f} | {delta_cell} |"
        )
    lines.append("")
    return lines


def _render_marker_section(
    summaries: dict[str, dict[str, str]],
    *,
    title: str,
    patterns: tuple,
    reflexion_key: str = REFLEXION_SUMMARY_KEY,
    meta_keys: tuple[str, ...] = META_SUMMARY_KEYS,
    report: ReportConfig,
) -> list[str]:
    story_count = len(summaries)
    lines = [
        f"Heuristic regex scan of integrated summaries ({len(patterns)} patterns).",
        "",
        "| Summary key | Questions flagged | Rate | vs reflexion-only |",
        "|-------------|-------------------|------|-------------------|",
    ]
    reflex_hits = stories_with_markers(summaries, reflexion_key, patterns)
    reflex_rate = len(reflex_hits) / story_count if story_count else 0.0
    lines.append(f"| {reflexion_key} | {len(reflex_hits)} | {reflex_rate:.0%} | — |")
    for meta_key in meta_keys:
        if meta_key not in next(iter(summaries.values()), {}):
            continue
        meta_hits = stories_with_markers(summaries, meta_key, patterns)
        added = meta_adds_markers_vs_reflexion(
            summaries, meta_key, reflexion_key=reflexion_key, patterns=patterns
        )
        rate = len(meta_hits) / story_count if story_count else 0.0
        lines.append(f"| {meta_key} | {len(meta_hits)} | {rate:.0%} | +{len(added)} vs reflexion |")
    lines.extend(["", f"### {title} — examples where meta adds markers reflexion missed", ""])
    for meta_key in meta_keys:
        if meta_key not in next(iter(summaries.values()), {}):
            continue
        added = meta_adds_markers_vs_reflexion(
            summaries, meta_key, reflexion_key=reflexion_key, patterns=patterns
        )
        if not added:
            lines.append(f"**{meta_key}:** no additional flagged questions vs `{reflexion_key}`.")
            lines.append("")
            continue
        lines.append(f"**{meta_key}** ({len(added)} questions):")
        lines.append("")
        for hit in added[: report.max_marker_examples]:
            lines.append(
                f"- `{hit.story_id[: report.id_display_chars]}` — {_escape_cell(hit.excerpt, report.excerpt_chars)}"
            )
        if len(added) > report.max_marker_examples:
            lines.append(f"- *…and {len(added) - report.max_marker_examples} more.*")
        lines.append("")
    return lines


def write_detailed_report(
    run: GPQABenchmarkRun,
    report_path: Path,
    *,
    baseline: str = "nano_direct",
    report: ReportConfig | None = None,
) -> Path:
    """Write standard report plus a detailed analysis markdown file."""
    # Presentation settings travel on the run, so a report regenerated later
    # looks the same as the one the run produced.
    report = report or getattr(run.config, "report", None) or ReportConfig()
    summary_path = report_path.with_name(report_path.name.replace("gpqa_detailed_", "gpqa_report_"))
    write_report(run, summary_path, report)

    metrics = compute_metrics(run.results)
    conditions = [c for c in run.config.conditions if c in metrics]
    if baseline in metrics and baseline not in conditions:
        conditions = [baseline, *conditions]

    lines: list[str] = [
        "# GPQA Diamond Detailed Benchmark Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Run configuration",
        "",
        f"- Questions: {run.config.num_questions} (seed {run.config.seed})",
        f"- Conditions: `{', '.join(run.config.conditions)}`",
        f"- Cache namespace: `{run.config.cache_namespace or '(legacy — no namespace)'}`",
        f"- Meta prompt profile: `{run.config.meta_prompt_profile}`",
        f"- Workers: {run.config.max_workers}",
        f"- Started: {run.started_at}",
        f"- Finished: {run.finished_at}",
        "",
        "## Executive summary",
        "",
    ]

    if not conditions:
        lines.append("No scored conditions in this run.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return report_path

    ranked = sorted(conditions, key=lambda c: metrics[c].accuracy, reverse=True)
    best = ranked[0]
    lines.append(
        f"Best condition: **{best}** at **{metrics[best].accuracy:.1%}** "
        f"({metrics[best].correct}/{metrics[best].total})."
    )
    if baseline in metrics:
        for cond in ranked:
            if cond == baseline:
                continue
            delta = metrics[cond].accuracy - metrics[baseline].accuracy
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"- `{cond}` vs `{baseline}`: {sign}{delta:.1%} "
                f"({metrics[cond].correct - metrics[baseline].correct:+d} questions)"
            )

    lines.extend(["", "## Aggregate accuracy", ""])
    lines.append("| Condition | Accuracy | Correct | Total |")
    lines.append("|-----------|----------|---------|-------|")
    for cond in ranked:
        m = metrics[cond]
        lines.append(f"| {cond} | {m.accuracy:.3f} | {m.correct} | {m.total} |")

    # High-level domain = Biology / Physics / Chemistry (GPQA metadata).
    # Skip when the mirror could not be labeled (all "unspecified").
    show_domains = has_labeled_domains(run.results)
    if show_domains:
        lines.extend(["", "## Accuracy by domain", ""])
        lines.extend(_type_table(metrics, ranked))
    else:
        lines.extend(
            [
                "",
                "## Accuracy by domain",
                "",
                "*Skipped — no High-level domain labels on this run "
                "(install pyarrow / refresh domain cache).*",
                "",
            ]
        )

    if baseline in metrics:
        lines.extend(["", f"## Pairwise comparison vs `{baseline}`", ""])
        lines.append("| Condition | Wins | Losses | Both correct | Both wrong | Net |")
        lines.append("|-----------|------|--------|--------------|------------|-----|")
        for cond in ranked:
            if cond == baseline:
                continue
            pw = _pairwise_vs_direct(run.results, cond)
            net = pw["wins"] - pw["losses"]
            lines.append(
                f"| {cond} | {pw['wins']} | {pw['losses']} | "
                f"{pw['both_correct']} | {pw['both_wrong']} | {net:+d} |"
            )

    if "nano_reflexion" in metrics and any(c.startswith("nano_meta") for c in metrics):
        lines.extend(["", "## Meta layer vs plain reflexion (depth 1)", ""])
        ref_acc = metrics["nano_reflexion"].accuracy
        for cond in ranked:
            if not cond.startswith("nano_meta"):
                continue
            m = metrics[cond]
            lines.append(
                f"- `{cond}`: {m.accuracy:.3f} vs `nano_reflexion` {ref_acc:.3f} "
                f"({m.accuracy - ref_acc:+.3f})"
            )

    meta_conds = [c for c in ranked if c.startswith("nano_meta")]
    if show_domains and baseline in metrics and meta_conds:
        lines.extend(["", "## Where meta helps or hurts (by domain)", ""])
        types = sorted({t for m in metrics.values() for t in m.by_type})
        for cond in meta_conds:
            lines.append(f"### `{cond}` vs `{baseline}`")
            lines.append("")
            lines.append("| Domain | Meta | Direct | Δ |")
            lines.append("|--------|------|--------|---|")
            for qtype in types:
                ma = metrics[cond].by_type.get(qtype, {}).get("accuracy")
                da = metrics[baseline].by_type.get(qtype, {}).get("accuracy")
                if ma is None or da is None:
                    continue
                lines.append(f"| {qtype} | {ma:.3f} | {da:.3f} | {ma - da:+.3f} |")
            lines.append("")

    compare_conditions = [
        c for c in ranked if c != baseline and c in {*run.config.conditions, *metrics}
    ]
    meta_conds = [c for c in compare_conditions if c.startswith("nano_meta")]
    reflex_conds = [
        c for c in compare_conditions if "reflexion" in c and not c.startswith("nano_meta")
    ]

    if baseline in metrics and compare_conditions:
        lines.extend(["", "## Question flips vs baseline", ""])
        lines.append(
            f"Baseline: `{baseline}`. **Wrong → right** = fixed; **right → wrong** = regression."
        )
        lines.append("")
        for cond in meta_conds + reflex_conds:
            flip = collect_flips(run.results, baseline=baseline, condition=cond)
            lines.extend(_render_flip_section(flip, report=report))

    if show_domains and baseline in metrics and compare_conditions:
        lines.extend(["", "## Domain error analysis", ""])
        lines.extend(
            _render_domain_error_section(
                run.results,
                baseline=baseline,
                conditions=meta_conds + reflex_conds,
            )
        )

    if run.summaries:
        lines.extend(["", "## Summary length (reasoning depth proxy)", ""])
        lines.extend(_render_summary_length_section(run.summaries))

        for marker_name, patterns in MARKER_SETS.items():
            title = marker_name.replace("_", " ").title()
            lines.extend(["", f"## Scientific-reasoning markers: {title}", ""])
            lines.extend(
                _render_marker_section(
                    run.summaries,
                    title=title,
                    patterns=patterns,
                    report=report,
                )
            )

    # Polycontextural dimension attribution (reuse OpenToM helpers via adapter).
    shared = as_opentom_run(run)
    meta_conds_for_attr = [c for c in ranked if c.startswith("nano_meta")]
    if baseline in metrics and meta_conds_for_attr and run.summaries:
        lines.extend(
            [
                "",
                "## Polycontextural dimension attribution",
                "",
                "Links **QA repairs** (baseline wrong → meta correct) to **judge pole "
                "repairs** between meta cycle 0 and the selected final cycle:",
                "",
                "- **Subjective:** F→A (inauthentic → authentic)",
                "- **Objective:** F→W (false → true)",
                "- **Dialectical:** R→W (redundant → productive)",
                "",
            ]
        )
        for cond in meta_conds_for_attr:
            summary_key = summary_key_for_condition(cond)
            if not summary_key:
                continue
            traces = load_meta_traces_for_run(shared, summary_key)
            attr = build_dimension_attribution(
                run.results,
                traces,
                baseline=baseline,
                condition=cond,
                summary_key=summary_key,
            )
            lines.append(f"### `{cond}` vs `{baseline}`")
            lines.append("")
            lines.append(
                f"- QA repairs: **{attr.repaired_questions}** questions across "
                f"**{attr.repaired_stories}** items"
            )
            lines.append(
                f"- Meta traces available: **{attr.stories_with_trace}** / "
                f"{attr.repaired_stories} repaired items "
                f"({attr.stories_without_trace} missing)"
            )
            if attr.stories_with_trace == 0:
                lines.append("")
                lines.append(
                    "*No `.meta.json` traces found beside cached summaries. "
                    "Re-run meta conditions (or backfill) then regenerate this report.*"
                )
                lines.append("")
                continue
            if attr.stories_no_dimension_repair:
                lines.append(
                    f"- Repaired items with **no** negative→positive judge flip: "
                    f"**{attr.stories_no_dimension_repair}**"
                )
            lines.append("")
            shares = repair_dimension_shares(attr)
            dominant = max(shares.items(), key=lambda kv: kv[1])
            lines.append(
                f"**Theory test:** among judge pole repairs tied to QA fixes, "
                f"**{dominant[0]}** carries **{dominant[1]:.0%}** of flips."
            )
            lines.append("")
            lines.extend(format_dimension_table(attr))
            lines.append("#### All label transitions (cycle 0 → final)")
            lines.append("")
            lines.extend(format_transition_table(attr))
            if attr.example_trajectories:
                lines.append("#### Example triple trajectories (repaired items)")
                lines.append("")
                lines.append("| ID | Repairs | Triple path | Termination | Judge repairs |")
                lines.append("|----|--------:|-------------|-------------|---------------|")
                for ex in attr.example_trajectories:
                    lines.append(
                        f"| {ex['story_id'][: report.id_display_chars]} | {ex['repaired_count']} | "
                        f"{ex['trajectory']} | {ex['termination']} | {ex['repair_dims']} |"
                    )
                lines.append("")

    charts = summary_path.parent / "charts"

    def rel(p: Path) -> str:
        """Chart paths in the report are relative to the report itself."""
        return p.relative_to(report_path.parent).as_posix()

    if charts.exists():
        lines.extend(
            [
                "## Charts",
                "",
                f"![Scores by condition]({rel(charts / 'gpqa_scores_by_condition.png')})",
                "",
            ]
        )
        if show_domains and (charts / "gpqa_scores_by_domain.png").exists():
            lines.extend(
                [
                    f"![Scores by domain]({rel(charts / 'gpqa_scores_by_domain.png')})",
                    "",
                ]
            )

    lines.extend(
        [
            "## Methodology notes",
            "",
            "- **Engine prompts:** `prompts_gpqa.json` (scientific reasoning summaries; "
            "no final letter in the summary).",
            "- **Meta profile `gpqa`:** `meta/prompts_gpqa.json` — judges and boundaries "
            "tuned for STEM constraints, mechanisms, and distractors.",
            "- **Meta profile `default`:** general `prompts_meta.json`.",
            "- **QA scoring:** exact A–D letter match (LLM judge only if letter ambiguous).",
            "- **Polycontextural judges:** gpt-4o (`meta_judge`).",
            f"- **Summary cache:** `results/cache/summaries/{run.config.cache_namespace or '<legacy>'}/`",
            "- **Domain labels:** GPQA High-level domain (Biology / Physics / Chemistry), "
            "from official CSV columns or nichenshun parquet metadata when using the "
            "public mirror.",
            "",
            f"Standard report: `{summary_path.name}`",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

"""Detailed markdown analysis for OpenToM benchmark runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from polyreflexion.benchmark.dimension_attribution import (
    build_dimension_attribution,
    format_dimension_table,
    format_transition_table,
    repair_dimension_shares,
)
from polyreflexion.benchmark.metrics import compute_metrics
from polyreflexion.benchmark.qualitative_analysis import (
    CONTRADICTION_PATTERNS,
    FALSE_BELIEF_PATTERNS,
    META_SUMMARY_KEYS,
    REFLEXION_SUMMARY_KEY,
    FlipRecord,
    FlipSummary,
    _escape_cell,
    collect_flips,
    count_by_type,
    inference_error_analysis,
    meta_adds_markers_vs_reflexion,
    stories_with_markers,
    summary_length_stats,
)
from polyreflexion.benchmark.report import write_report
from polyreflexion.benchmark.runner import (
    BenchmarkRun,
    load_meta_traces_for_run,
    summary_key_for_condition,
)


def _pairwise_vs_direct(results: list[dict], condition: str) -> dict[str, int]:
    """Count wins/losses vs nano_direct on the same questions."""
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
        "| Question type | " + " | ".join(conditions) + " |",
        "|---------------|" + "|".join(["---"] * len(conditions)) + "|",
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
    """Markdown bullet list of flip counts by question type."""
    if not records:
        return ["- *(none)*"]
    by_type = count_by_type(records)
    return [f"- `{qtype}`: {count}" for qtype, count in by_type.most_common()]


def _flip_examples_table(records: list[FlipRecord], *, max_rows: int = 12) -> list[str]:
    """Compact table of individual flip examples."""
    if not records:
        return ["*(none)*", ""]
    lines = [
        "| Story | Type | Gold | Baseline | Condition | Question |",
        "|-------|------|------|----------|-----------|----------|",
    ]
    for rec in records[:max_rows]:
        lines.append(
            f"| {rec.story_id[:8]} | {rec.question_type} | {rec.gold_label} | "
            f"{rec.baseline_pred} | {rec.condition_pred} | "
            f"{_escape_cell(rec.question, 60)} |"
        )
    if len(records) > max_rows:
        lines.append("")
        lines.append(f"*…and {len(records) - max_rows} more.*")
    lines.append("")
    return lines


def _render_flip_section(flip: FlipSummary, *, max_examples: int = 12) -> list[str]:
    """Markdown for one condition's gain/loss breakdown."""
    lines = [
        f"### `{flip.condition}` vs `{flip.baseline}` "
        f"(net {flip.net:+d}: {len(flip.gains)} fixed, {len(flip.losses)} regressed)",
        "",
        "#### Wrong → right",
        "",
        "By question type:",
        *_flip_type_breakdown(flip.gains),
        "",
        *_flip_examples_table(flip.gains, max_rows=max_examples),
        "#### Right → wrong",
        "",
        "By question type:",
        *_flip_type_breakdown(flip.losses),
        "",
        *_flip_examples_table(flip.losses, max_rows=max_examples),
    ]
    return lines


def _render_inference_section(
    results: list[dict],
    *,
    baseline: str,
    conditions: list[str],
) -> list[str]:
    lines = [
        "Counts are on **location-*** and **multihop-*** question types only "
        "(faulty ToM inference proxy; not free-text hallucination in answers).",
        "",
        "| Condition | Baseline wrong | Condition wrong | Fixed | New errors | "
        "Both wrong (same ans.) | Both wrong (changed) | Δ error rate |",
        "|-----------|----------------|-----------------|-------|------------|"
        "----------------------|----------------------|--------------|",
    ]
    base_stats = inference_error_analysis(results, baseline=baseline, condition=baseline)
    base_rate = base_stats["baseline_error_rate"]
    for cond in conditions:
        if cond == baseline:
            continue
        stats = inference_error_analysis(results, baseline=baseline, condition=cond)
        delta = stats["condition_error_rate"] - base_rate
        lines.append(
            f"| {cond} | {stats['baseline_wrong']} | {stats['condition_wrong']} | "
            f"{stats['fixed_from_baseline']} | {stats['new_errors_vs_baseline']} | "
            f"{stats['both_wrong_same_answer']} | {stats['both_wrong_different_answer']} | "
            f"{delta:+.1%} |"
        )
    lines.extend(
        [
            "",
            "**Reading this table:** *Fixed* = baseline wrong, condition correct (fewer inference errors). "
            "*New errors* = baseline correct, condition wrong (possible new confabulation). "
            "*Both wrong (same ans.)* = persistent wrong belief; meta did not change the mistaken label.",
            "",
        ]
    )
    return lines


def _render_summary_length_section(
    summaries: dict[str, dict[str, str]],
    *,
    reflexion_key: str = REFLEXION_SUMMARY_KEY,
) -> list[str]:
    keys = [reflexion_key, *META_SUMMARY_KEYS]
    stats = [summary_length_stats(summaries, key) for key in keys]
    stats = [s for s in stats if s is not None]
    if not stats:
        return ["No story summaries available in this run.", ""]
    base = next((s for s in stats if s.key == reflexion_key), stats[0])
    lines = [
        "Story summaries are the context fed to the QA model (proxy for integrated reasoning length).",
        "",
        "| Summary key | Stories | Mean words | Mean chars | vs reflexion (words) |",
        "|-------------|---------|------------|------------|----------------------|",
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
    max_examples: int = 8,
) -> list[str]:
    story_count = len(summaries)
    lines = [
        f"Heuristic regex scan of integrated summaries ({len(patterns)} patterns). "
        "This proxies explicit awareness in text, not meta-cycle trace logs.",
        "",
        "| Summary key | Stories flagged | Rate | vs reflexion-only |",
        "|-------------|-----------------|------|-------------------|",
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
        lines.append(
            f"| {meta_key} | {len(meta_hits)} | {rate:.0%} | +{len(added)} stories vs reflexion |"
        )
    lines.extend(["", f"### {title} — examples where meta adds markers reflexion missed", ""])
    for meta_key in meta_keys:
        if meta_key not in next(iter(summaries.values()), {}):
            continue
        added = meta_adds_markers_vs_reflexion(
            summaries, meta_key, reflexion_key=reflexion_key, patterns=patterns
        )
        if not added:
            lines.append(f"**{meta_key}:** no additional flagged stories vs `{reflexion_key}`.")
            lines.append("")
            continue
        lines.append(f"**{meta_key}** ({len(added)} stories):")
        lines.append("")
        for hit in added[:max_examples]:
            lines.append(f"- `{hit.story_id[:8]}` — {_escape_cell(hit.excerpt, 100)}")
        if len(added) > max_examples:
            lines.append(f"- *…and {len(added) - max_examples} more.*")
        lines.append("")
    return lines


def write_detailed_report(
    run: BenchmarkRun,
    report_path: Path,
    *,
    baseline: str = "nano_direct",
) -> Path:
    """Write standard report plus a detailed analysis markdown file."""
    # Standard charts + summary table land alongside the detailed doc.
    summary_path = report_path.with_name(
        report_path.name.replace("opentom_detailed_", "opentom_report_")
    )
    write_report(run, summary_path)

    metrics = compute_metrics(run.results)
    conditions = [c for c in run.config.conditions if c in metrics]
    if baseline in metrics and baseline not in conditions:
        conditions = [baseline, *conditions]

    lines: list[str] = [
        "# OpenToM Detailed Benchmark Report",
        "",
        f"Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Run configuration",
        "",
        f"- Stories: {run.config.num_stories} (seed {run.config.seed})",
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

    lines.extend(["", "## Accuracy by question type", ""])
    lines.extend(_type_table(metrics, ranked))

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

    # Meta vs reflexion d1 when both present
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

    # Per-type delta meta c2 vs direct
    meta_conds = [c for c in ranked if c.startswith("nano_meta")]
    if baseline in metrics and meta_conds:
        lines.extend(["", "## Where meta helps or hurts (by question type)", ""])
        types = sorted({t for m in metrics.values() for t in m.by_type})
        for cond in meta_conds:
            lines.append(f"### `{cond}` vs `{baseline}`")
            lines.append("")
            lines.append("| Type | Meta | Direct | Δ |")
            lines.append("|------|------|--------|---|")
            for qtype in types:
                ma = metrics[cond].by_type.get(qtype, {}).get("accuracy")
                da = metrics[baseline].by_type.get(qtype, {}).get("accuracy")
                if ma is None or da is None:
                    continue
                lines.append(f"| {qtype} | {ma:.3f} | {da:.3f} | {ma - da:+.3f} |")
            lines.append("")

    # --- Qualitative deep dives ---
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
        # Meta first, then reflexion variants.
        for cond in meta_conds + reflex_conds:
            flip = collect_flips(run.results, baseline=baseline, condition=cond)
            lines.extend(_render_flip_section(flip))

    if baseline in metrics and compare_conditions and run.summaries:
        lines.extend(["", "## Does meta reduce ToM inference errors?", ""])
        lines.extend(
            _render_inference_section(
                run.results,
                baseline=baseline,
                conditions=meta_conds + reflex_conds,
            )
        )

        lines.extend(["", "## Summary length (reasoning depth proxy)", ""])
        lines.extend(_render_summary_length_section(run.summaries))

        lines.extend(
            [
                "",
                "## False-belief awareness in summaries",
                "",
            ]
        )
        lines.extend(
            _render_marker_section(
                run.summaries,
                title="False-belief markers",
                patterns=FALSE_BELIEF_PATTERNS,
            )
        )

        lines.extend(
            [
                "",
                "## Contradiction and belief–reality tension",
                "",
                "OpenToM meta runs use a contradiction check on negation boundaries during "
                "summary generation, but those traces are not persisted in benchmark JSON. "
                "Below: heuristic scan of final summary text for conflict / gap language.",
                "",
            ]
        )
        lines.extend(
            _render_marker_section(
                run.summaries,
                title="Contradiction / tension markers",
                patterns=CONTRADICTION_PATTERNS,
            )
        )

    # --- Polycontextural dimension attribution (theory-linked) ---
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
                "Cycle 0 `(A,W,W)` is reinterpreted to `(A,W,R)` by design; that dialectical "
                "W→R transition is included in the full transition table below.",
                "",
            ]
        )
        for cond in meta_conds_for_attr:
            summary_key = summary_key_for_condition(cond)
            if not summary_key:
                continue
            traces = load_meta_traces_for_run(run, summary_key)
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
                f"**{attr.repaired_stories}** stories"
            )
            lines.append(
                f"- Meta traces available: **{attr.stories_with_trace}** / "
                f"{attr.repaired_stories} repaired stories "
                f"({attr.stories_without_trace} missing — run "
                f"`python backfill_meta_traces.py <opentom_run.json>`)"
            )
            if attr.stories_with_trace == 0:
                lines.append("")
                lines.append(
                    "*No `.meta.json` traces found beside cached summaries. "
                    "Backfill traces, then regenerate this report.*"
                )
                lines.append("")
                continue
            if attr.stories_no_dimension_repair:
                lines.append(
                    f"- Repaired stories with **no** negative→positive judge flip: "
                    f"**{attr.stories_no_dimension_repair}** "
                    "(QA gain without a pole repair on cycle 0→final)"
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
                lines.append("#### Example triple trajectories (repaired stories)")
                lines.append("")
                lines.append("| Story | Repairs | Triple path | Termination | Judge repairs |")
                lines.append("|-------|--------:|-------------|-------------|---------------|")
                for ex in attr.example_trajectories:
                    lines.append(
                        f"| {ex['story_id'][:8]} | {ex['repaired_count']} | "
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
                f"![Scores by condition]({rel(charts / 'scores_by_condition.png')})",
                "",
                f"![Scores by question type]({rel(charts / 'scores_by_question_type.png')})",
                "",
            ]
        )

    lines.extend(
        [
            "## Methodology notes",
            "",
            "- **Engine prompts:** OpenToM `prompts.json` (integrated story summaries).",
            "- **Meta profile `opentom`:** `meta/prompts_opentom.json` — judges and boundaries "
            "tuned for false beliefs, locations, and witness structure.",
            "- **Meta profile `default`:** general `prompts_meta.json`.",
            "- **QA judge:** gpt-5-mini (OpenToM label match).",
            "- **Polycontextural judges:** gpt-4o (`meta_judge`).",
            f"- **Summary cache:** `results/cache/summaries/{run.config.cache_namespace or '<legacy>'}/`",
            "",
            f"Standard report: `{summary_path.name}`",
        ]
    )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path

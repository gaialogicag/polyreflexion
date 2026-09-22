"""JSONL trace recording and markdown reporting for meta runs.

Layout per run directory (``results/meta/<run_id>/``):

- ``trace.jsonl``  — one line per cycle (triple, rationales, geometry, action)
- ``run.json``     — the full ``MetaResult`` as one JSON document
- ``report.md``    — human-readable summary table + final answer + topology images
"""

from __future__ import annotations

import json
from pathlib import Path

from polyrx.meta.datatypes import MetaCycle, MetaResult


class MetaTrace:
    """Write cycle records and the final run artifacts to one run directory."""

    def __init__(self, run_dir: Path | str) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._trace_path = self.run_dir / "trace.jsonl"
        self._recorded = 0

    def record(self, cycle: MetaCycle) -> None:
        """Append one cycle to the JSONL trace (usable while the run is live)."""
        with self._trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(cycle.to_dict(), ensure_ascii=False) + "\n")
        self._recorded += 1

    def save(self, result: MetaResult) -> Path:
        """Persist the complete run: trace.jsonl, run.json, and report.md."""
        # Rewrite the trace from the result so save() is correct even when
        # record() was never called incrementally.
        with self._trace_path.open("w", encoding="utf-8") as handle:
            for cycle in result.cycles:
                handle.write(json.dumps(cycle.to_dict(), ensure_ascii=False) + "\n")
        self._recorded = len(result.cycles)

        run_path = self.run_dir / "run.json"
        run_path.write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        report_path = self.run_dir / "report.md"
        report_path.write_text(self._render_report(result), encoding="utf-8")
        return report_path

    def _render_report(self, result: MetaResult) -> str:
        lines: list[str] = [
            "# Meta-Reflexion Run",
            "",
            f"**Question:** {result.question}",
            "",
            f"**Termination:** `{result.termination_reason}` — "
            f"selected cycle **{result.selected_cycle}**",
            "",
            "## Cycles",
            "",
            "| Cycle | Triple | Global | Action | Drifted | Boundaries | Path |",
            "|------:|--------|:------:|--------|:-------:|-----------:|------|",
        ]
        for cycle in result.cycles:
            path = " → ".join(cycle.topology.path) or "root"
            lines.append(
                f"| {cycle.index} | `{cycle.evaluation.as_triple()}` "
                f"| {cycle.global_value.value} | {cycle.action or '—'} "
                f"| {'yes' if cycle.drifted else 'no'} "
                f"| {len(cycle.boundaries)} | {path} |"
            )

        lines += ["", "## Final answer", "", result.final_answer or "*(no answer)*", ""]

        # Embed topology renderings when meta/viz.py has produced them.
        images = sorted(self.run_dir.glob("topology_*.png"))
        if images:
            lines.append("## Topology")
            lines.append("")
            for image in images:
                lines.append(f"![{image.stem}]({image.name})")
            lines.append("")

        lines.append("## Judge rationales")
        lines.append("")
        for cycle in result.cycles:
            lines.append(f"### Cycle {cycle.index} — `{cycle.evaluation.as_triple()}`")
            for judgment in cycle.evaluation.judgments():
                lines.append(
                    f"- **{judgment.dimension.value}** ({judgment.label}, "
                    f"{judgment.method}): {judgment.rationale}"
                )
            lines.append("")
        return "\n".join(lines)

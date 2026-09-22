"""GPQA Diamond: graduate-level biology, physics and chemistry questions.

Needs its own adapter for two reasons the tabular one cannot cover:

* the official release and the public mirror have different shapes, and which
  one answers depends on whether the user holds a licence for the gated repo,
  so a single column mapping does not describe the source;
* the mirror ships no domain labels, which are recovered by matching question
  stems against a third metadata repository.
"""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

from polyrx.datasets.base import DatasetAdapter, Item, register_adapter
from polyrx.datasets.tabular import CHOICE_LETTERS, TabularAdapter

#: Cached stem -> domain map recovered from the metadata repository.
DOMAIN_CACHE_NAME = "domain_by_stem.json"

_MIRROR_CHOICE_RE = re.compile(r"^\(([A-D])\)\s*(.+)$", re.MULTILINE)
_STEM_SPLIT_RE = re.compile(r"\n(?:Choices|Answer Choices)\s*:\s*\n", re.IGNORECASE)


def normalize_stem(text: str) -> str:
    """Normalise a question stem so the same question matches across mirrors."""
    stem = _STEM_SPLIT_RE.split(text, maxsplit=1)[0]
    return re.sub(r"\s+", " ", stem).strip().lower()


@register_adapter
class GPQAAdapter(DatasetAdapter):
    name = "gpqa"

    def load(self, path: Path) -> list[Item]:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
        if not rows:
            raise ValueError(f"No rows in {path}")

        domains = self._domain_map(path.parent)
        tabular = TabularAdapter(self.config)
        items: list[Item] = []
        for row in rows:
            if self._is_official(row):
                items.append(self._from_official(row, tabular))
            elif self._is_mirror(row):
                items.append(self._from_mirror(row, domains))
        if not items:
            raise ValueError(
                f"{path} matched neither the official nor the mirror GPQA layout. "
                f"Columns: {sorted(rows[0])}"
            )
        return items

    # -- the two source layouts --------------------------------------------

    @staticmethod
    def _is_official(row: dict[str, str]) -> bool:
        return "Question" in row and "Correct Answer" in row and "Incorrect Answer 1" in row

    @staticmethod
    def _is_mirror(row: dict[str, str]) -> bool:
        return "problem" in row and "answer" in row

    def _from_official(self, row: dict[str, str], tabular: TabularAdapter) -> Item:
        stem = (row.get("Question") or "").strip()
        correct = (row.get("Correct Answer") or "").strip()
        wrong = [(row.get(f"Incorrect Answer {n}") or "").strip() for n in (1, 2, 3)]
        if not stem or not correct or not all(wrong):
            raise ValueError("Official GPQA row is missing a question or an answer")

        # Reuse the shared deterministic shuffle so choice order matches the
        # tabular adapter's, and stays stable across runs.
        options = tabular._choices(
            {"_c": correct, "_w1": wrong[0], "_w2": wrong[1], "_w3": wrong[2]},
            stem,
            correct,
        ) or [correct, *wrong]
        gold = next(
            letter for letter, text in zip(CHOICE_LETTERS, options, strict=False) if text == correct
        )
        domain = (
            (row.get("High-level domain") or row.get("High-level Domain") or "").strip()
            or (row.get("Subdomain") or "").strip()
            or "unspecified"
        )
        return Item(
            item_id=Item.content_id(stem, correct, *wrong, id_chars=self.config.id_hash_chars),
            context=self._render(stem, options),
            prompt_text=self._render(stem, options),
            question=stem,
            gold_label=gold,
            label_space=", ".join(CHOICE_LETTERS[: len(options)]),
            group=domain,
            metadata={"choices": dict(zip(CHOICE_LETTERS, options, strict=False))},
        )

    def _from_mirror(self, row: dict[str, str], domains: dict[str, str]) -> Item:
        problem = (row.get("problem") or "").strip()
        answer = (row.get("answer") or "").strip().upper()
        if answer not in CHOICE_LETTERS[:4]:
            raise ValueError(f"Mirror GPQA answer must be A-D, got {answer!r}")
        stem, choices = self._parse_mirror(problem)
        domain = domains.get(normalize_stem(stem)) or domains.get(normalize_stem(problem))
        return Item(
            item_id=Item.content_id(problem, answer, id_chars=self.config.id_hash_chars),
            context=problem,
            prompt_text=problem if "Choices:" in problem else self._render(stem, choices),
            question=stem,
            gold_label=answer,
            label_space=", ".join(CHOICE_LETTERS[:4]),
            group=domain or "unspecified",
            metadata={"choices": dict(zip(CHOICE_LETTERS, choices, strict=False))},
        )

    @staticmethod
    def _parse_mirror(problem: str) -> tuple[str, list[str]]:
        match = re.search(r"\nChoices:\s*\n", problem)
        if not match:
            raise ValueError("Mirror GPQA problem has no 'Choices:' block")
        stem = problem[: match.start()].strip()
        found = dict(_MIRROR_CHOICE_RE.findall(problem[match.end() :]))
        if set(found) != set(CHOICE_LETTERS[:4]):
            raise ValueError(f"Expected choices A-D, got {sorted(found)}")
        return stem, [found[letter].strip() for letter in CHOICE_LETTERS[:4]]

    @staticmethod
    def _render(stem: str, choices: list[str]) -> str:
        lines = [stem.strip(), "", "Choices:"]
        lines += [
            f"({letter}) {text}" for letter, text in zip(CHOICE_LETTERS, choices, strict=False)
        ]
        return "\n".join(lines)

    # -- domain labels ------------------------------------------------------

    def _domain_map(self, cache_dir: Path) -> dict[str, str]:
        """Stem -> domain, from the metadata repository. Optional by design.

        The public mirror carries no domain labels. Without them every item
        groups as ``unspecified``, which costs the per-domain breakdown and
        nothing else, so a failure here is reported rather than raised.
        """
        cached = cache_dir / DOMAIN_CACHE_NAME
        if cached.is_file():
            return json.loads(cached.read_text(encoding="utf-8"))
        if not self.config.extras:
            return {}

        try:
            import pyarrow.parquet as pq

            from polyrx.data.fetch import DatasetFetcher

            fetcher = DatasetFetcher(cache_dir.parent, verify=self.config.verify_checksums)
            parquet = fetcher.fetch(self.config.extras[0], subdir=f"{self.config.name}/metadata")
            table = pq.read_table(parquet)
        except Exception as exc:
            print(
                f"GPQA domain labels unavailable ({type(exc).__name__}); grouping as unspecified."
            )
            return {}

        mapping: dict[str, str] = {}
        for i in range(table.num_rows):
            raw = table.column("metadata")[i].as_py()
            try:
                meta = ast.literal_eval(raw) if isinstance(raw, str) else (raw or {})
            except (SyntaxError, ValueError):
                continue
            domain = str(meta.get("High-level domain") or "").strip()
            if not domain:
                continue
            for key in (table.column("question")[i].as_py(), meta.get("Pre-Revision Question")):
                if key:
                    mapping[normalize_stem(str(key))] = domain

        cached.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Cached {len(mapping)} GPQA domain keys -> {cached}")
        return mapping

    # -- scoring ------------------------------------------------------------

    def match_label(self, prediction: str, item: Item) -> str:
        return TabularAdapter(self.config).match_label(prediction, item)

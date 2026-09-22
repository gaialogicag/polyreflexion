"""Load and sample the GPQA Diamond graduate-level Q&A benchmark.

Supports two CSV shapes:

1. **Official** (``Idavidrein/gpqa``) — separate correct / incorrect answers,
   domain metadata; choices are shuffled with a fixed seed.
2. **Public mirror** (``aradhye/gpqa_diamond``) — ``problem`` + letter ``answer``
   with choices already embedded. High-level domains (Biology / Physics /
   Chemistry) are recovered from the public ``nichenshun/gpqa_diamond`` parquet
   metadata when available.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from polyreflexion.config import DatasetConfig, DatasetFile
from polyreflexion.data.fetch import DatasetFetcher

#: Default on-disk location; override through ``PathsConfig.data_dir``.
CACHE_DIR = Path("data") / "gpqa"

#: Official release first (domain metadata plus raw answer texts), public
#: mirror second. The official repo is gated: without an accepted licence and
#: an ``HF_TOKEN`` the fetch falls through to the mirror, and domain labels are
#: then recovered from a third repo listed under ``extras``.
GPQA_DATASET = DatasetConfig(
    name="gpqa",
    primary=DatasetFile(
        repo_id="Idavidrein/gpqa", filename="gpqa_diamond.csv", gated=True
    ),
    fallbacks=[DatasetFile(repo_id="aradhye/gpqa_diamond", filename="gpqa_diamond.csv")],
    extras=[
        DatasetFile(
            repo_id="nichenshun/gpqa_diamond",
            filename="data/train-00000-of-00001.parquet",
        )
    ],
)
LABEL_SPACE = "A, B, C, D"
_CHOICE_LETTERS = ("A", "B", "C", "D")
DOMAIN_CACHE_NAME = "domain_by_stem.json"

# Mirror problems look like: "...\nChoices:\n(A) ...\n(B) ..."
_MIRROR_CHOICE_RE = re.compile(
    r"^\(([A-D])\)\s*(.+)$",
    re.MULTILINE,
)
_STEM_SPLIT_RE = re.compile(
    r"\n(?:Choices|Answer Choices)\s*:\s*\n",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GPQAItem:
    """One GPQA Diamond multiple-choice question."""

    question_id: str
    prompt_text: str
    question: str
    gold_label: str
    question_type: str
    label_space: str
    choices: dict[str, str]


def _question_id(*parts: str) -> str:
    blob = "\n".join(parts)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _format_prompt(stem: str, choices: dict[str, str]) -> str:
    lines = [stem.strip(), "", "Choices:"]
    for letter in _CHOICE_LETTERS:
        lines.append(f"({letter}) {choices[letter]}")
    return "\n".join(lines)


def _parse_mirror_problem(problem: str) -> tuple[str, dict[str, str]]:
    """Split mirror ``problem`` text into stem and A–D choices."""
    match = re.search(r"\nChoices:\s*\n", problem)
    if not match:
        raise ValueError("Mirror GPQA problem missing 'Choices:' block")
    stem = problem[: match.start()].strip()
    choice_block = problem[match.end() :]
    choices: dict[str, str] = {}
    for letter, text in _MIRROR_CHOICE_RE.findall(choice_block):
        choices[letter] = text.strip()
    if set(choices) != set(_CHOICE_LETTERS):
        raise ValueError(f"Expected choices A–D, got {sorted(choices)}")
    return stem, choices


def normalize_stem(text: str) -> str:
    """Normalize a question stem for domain lookup across dataset mirrors."""
    stem = _STEM_SPLIT_RE.split(text, maxsplit=1)[0]
    stem = re.sub(r"\s+", " ", stem).strip().lower()
    return stem


def _download_domain_map(cache_dir: Path, dataset: DatasetConfig | None = None) -> dict[str, str]:
    """Map normalized stems → High-level domain via nichenshun parquet metadata."""
    cache_path = cache_dir / DOMAIN_CACHE_NAME
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print(
            "pyarrow not installed; mirror items keep domain='unspecified'. "
            "Install pyarrow to recover Biology/Physics/Chemistry labels."
        )
        return {}

    dataset = dataset or GPQA_DATASET
    if not dataset.extras:
        return {}
    try:
        fetcher = DatasetFetcher(cache_dir.parent, verify=dataset.verify_checksums)
        parquet_path = fetcher.fetch(dataset.extras[0], subdir=f"{dataset.name}/nichenshun")
    except Exception as exc:
        print(f"Domain metadata download failed ({type(exc).__name__}); domains unspecified.")
        return {}

    table = pq.read_table(parquet_path)
    domain_map: dict[str, str] = {}
    for i in range(table.num_rows):
        question = table.column("question")[i].as_py() or ""
        meta_raw = table.column("metadata")[i].as_py()
        try:
            meta = ast.literal_eval(meta_raw) if isinstance(meta_raw, str) else (meta_raw or {})
        except (SyntaxError, ValueError):
            meta = {}
        domain = str(meta.get("High-level domain") or "").strip()
        if not domain:
            continue
        key = normalize_stem(question)
        if key:
            domain_map[key] = domain
        # Official pre-revision stem is often a closer match to the mirror CSV.
        pre = str(meta.get("Pre-Revision Question") or "").strip()
        if pre:
            domain_map[normalize_stem(pre)] = domain

    cache_path.write_text(json.dumps(domain_map, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Cached {len(domain_map)} GPQA domain stem keys → {cache_path}")
    return domain_map


def _is_official_row(row: dict[str, str]) -> bool:
    return (
        "Question" in row
        and "Correct Answer" in row
        and "Incorrect Answer 1" in row
        and "Incorrect Answer 2" in row
        and "Incorrect Answer 3" in row
    )


def _is_mirror_row(row: dict[str, str]) -> bool:
    return "problem" in row and "answer" in row


def _item_from_official(row: dict[str, str], *, shuffle_seed: int) -> GPQAItem:
    stem = (row.get("Question") or "").strip()
    correct = (row.get("Correct Answer") or "").strip()
    incorrect = [
        (row.get("Incorrect Answer 1") or "").strip(),
        (row.get("Incorrect Answer 2") or "").strip(),
        (row.get("Incorrect Answer 3") or "").strip(),
    ]
    if not stem or not correct or any(not x for x in incorrect):
        raise ValueError("Official GPQA row missing question or answers")

    options = [correct, *incorrect]
    # Deterministic per-question shuffle so merge/extend runs stay stable.
    rng = random.Random(
        int(hashlib.sha256(f"{shuffle_seed}:{stem}:{correct}".encode()).hexdigest()[:8], 16)
    )
    rng.shuffle(options)
    choices = {letter: text for letter, text in zip(_CHOICE_LETTERS, options, strict=True)}
    gold = next(letter for letter, text in choices.items() if text == correct)

    domain = (
        (row.get("High-level domain") or row.get("High-level Domain") or "").strip()
        or (row.get("Subdomain") or "").strip()
        or "unspecified"
    )
    qid = _question_id(stem, correct, *incorrect)
    return GPQAItem(
        question_id=qid,
        prompt_text=_format_prompt(stem, choices),
        question=stem,
        gold_label=gold,
        question_type=domain,
        label_space=LABEL_SPACE,
        choices=choices,
    )


def _item_from_mirror(
    row: dict[str, str],
    *,
    domain_map: dict[str, str] | None = None,
) -> GPQAItem:
    problem = (row.get("problem") or "").strip()
    answer = (row.get("answer") or "").strip().upper()
    if answer not in _CHOICE_LETTERS:
        raise ValueError(f"Mirror GPQA answer must be A–D, got {answer!r}")
    stem, choices = _parse_mirror_problem(problem)
    qid = _question_id(problem, answer)
    domain = "unspecified"
    if domain_map:
        domain = domain_map.get(normalize_stem(stem)) or domain_map.get(
            normalize_stem(problem)
        ) or "unspecified"
    return GPQAItem(
        question_id=qid,
        prompt_text=problem if "Choices:" in problem else _format_prompt(stem, choices),
        question=stem,
        gold_label=answer,
        question_type=domain,
        label_space=LABEL_SPACE,
        choices=choices,
    )


def download_gpqa_diamond(
    cache_dir: Path | None = None,
    *,
    dataset: DatasetConfig | None = None,
) -> Path:
    """Fetch the diamond CSV at its pinned revision: official, else mirror.

    Both sources publish the file under the same name, so the caller sees one
    stable path regardless of which one answered. Which source was used is
    recorded in the run's provenance block.
    """
    cache_dir = Path(cache_dir or CACHE_DIR)
    dataset = dataset or GPQA_DATASET
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / "gpqa_diamond.csv"
    if target.exists():
        return target

    fetcher = DatasetFetcher(cache_dir.parent, verify=dataset.verify_checksums)
    downloaded = fetcher.fetch_dataset(dataset)
    if downloaded.resolve() != target.resolve():
        target.write_text(downloaded.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def load_items(
    cache_dir: Path | None = None,
    *,
    shuffle_seed: int = 42,
) -> list[GPQAItem]:
    """Load all GPQA Diamond items from the cached CSV."""
    cache_dir = cache_dir or CACHE_DIR
    path = download_gpqa_diamond(cache_dir)
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows in {path}")

    items: list[GPQAItem] = []
    if _is_official_row(rows[0]):
        for row in rows:
            items.append(_item_from_official(row, shuffle_seed=shuffle_seed))
    elif _is_mirror_row(rows[0]):
        domain_map = _download_domain_map(cache_dir)
        for row in rows:
            items.append(_item_from_mirror(row, domain_map=domain_map))
        labeled = sum(1 for item in items if item.question_type != "unspecified")
        print(f"GPQA domains labeled: {labeled}/{len(items)}")
    else:
        raise ValueError(
            f"Unrecognized GPQA CSV columns: {list(rows[0].keys())}. "
            "Expected official Idavidrein/gpqa or aradhye/gpqa_diamond mirror."
        )
    return items


def domain_by_question_id(
    items: list[GPQAItem] | None = None,
    *,
    cache_dir: Path | None = None,
    shuffle_seed: int = 42,
) -> dict[str, str]:
    """Return ``question_id → High-level domain`` for enriching saved runs."""
    if items is None:
        items = load_items(cache_dir, shuffle_seed=shuffle_seed)
    return {item.question_id: item.question_type for item in items}


def sample_questions(
    items: list[GPQAItem],
    *,
    num_questions: int = 10,
    seed: int = 42,
) -> list[GPQAItem]:
    """Return a fixed random sample of questions."""
    ids = sorted({item.question_id for item in items})
    by_id = {item.question_id: item for item in items}
    rng = random.Random(seed)
    chosen = rng.sample(ids, min(num_questions, len(ids)))
    return [by_id[qid] for qid in chosen]


def extend_question_sample(
    items: list[GPQAItem],
    exclude_question_ids: set[str],
    *,
    num_additional: int,
    seed: int = 42,
) -> list[GPQAItem]:
    """Sample additional questions not in ``exclude_question_ids``."""
    by_id = {item.question_id: item for item in items}
    remaining = sorted(qid for qid in by_id if qid not in exclude_question_ids)
    rng = random.Random(seed)
    chosen = rng.sample(remaining, min(num_additional, len(remaining)))
    return [by_id[qid] for qid in chosen]

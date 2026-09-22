"""Load and sample the OpenToM Theory-of-Mind benchmark."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

from polyreflexion.config import DatasetConfig, DatasetFile
from polyreflexion.data.fetch import DatasetFetcher

#: Default on-disk location; override through ``PathsConfig.data_dir``.
CACHE_DIR = Path("data") / "opentom"

#: Where OpenToM comes from. The revision is pinned in ``data/MANIFEST.json``
#: and by ``conf/dataset/opentom.yaml``; this default only covers direct calls.
OPENTOM_DATASET = DatasetConfig(
    name="opentom",
    primary=DatasetFile(repo_id="SeacowX/OpenToM", filename="opentom.json"),
)


@dataclass(frozen=True)
class OpenToMItem:
    """One OpenToM question with story context."""

    story_id: str
    narrative: str
    question: str
    gold_label: str
    question_type: str
    label_space: str
    plot_info: dict


def _story_id(narrative: str) -> str:
    return hashlib.sha256(narrative.encode()).hexdigest()[:12]


def infer_label_space(question: str, question_type: str, plot_info: dict) -> str:
    """Derive allowed labels from question type and story metadata."""
    q_lower = question.lower()
    if question_type == "attitude":
        return "positive, negative, neutral"
    if "accessible" in q_lower:
        return "more accessible, equally accessible, less accessible"
    if "full" in q_lower:
        return "more full, equally full, less full"
    if "initial location" in q_lower:
        return "Yes, No"
    places = [
        plot_info.get("original_place", ""),
        plot_info.get("move_to_place", ""),
    ]
    unique = []
    for place in places:
        if place and place not in unique:
            unique.append(place)
    if unique:
        return ", ".join(unique)
    return "Yes, No"


def download_opentom(
    cache_dir: Path | None = None,
    *,
    dataset: DatasetConfig | None = None,
) -> Path:
    """Fetch ``opentom.json`` at its pinned revision, verifying the checksum."""
    cache_dir = Path(cache_dir or CACHE_DIR)
    dataset = dataset or OPENTOM_DATASET
    target = cache_dir / dataset.primary.filename
    if target.exists():
        return target
    fetcher = DatasetFetcher(cache_dir.parent, verify=dataset.verify_checksums)
    return fetcher.fetch_dataset(dataset)


def load_items(
    cache_dir: Path | None = None,
    *,
    dataset: DatasetConfig | None = None,
) -> list[OpenToMItem]:
    """Load all OpenToM QA items from the cached dataset."""
    path = download_opentom(cache_dir, dataset=dataset)
    raw = json.loads(path.read_text(encoding="utf-8"))
    items: list[OpenToMItem] = []
    for entry in raw:
        q = entry["question"]
        plot_info = entry.get("plot_info", {})
        narrative = entry["narrative"]
        question_text = q["question"]
        qtype = q["type"]
        items.append(
            OpenToMItem(
                story_id=_story_id(narrative),
                narrative=narrative,
                question=question_text,
                gold_label=q["answer"],
                question_type=qtype,
                label_space=infer_label_space(question_text, qtype, plot_info),
                plot_info=plot_info,
            )
        )
    return items


def sample_stories(
    items: list[OpenToMItem],
    *,
    num_stories: int = 10,
    seed: int = 42,
) -> list[OpenToMItem]:
    """Return all questions belonging to a fixed random sample of stories."""
    by_story: dict[str, list[OpenToMItem]] = {}
    for item in items:
        by_story.setdefault(item.story_id, []).append(item)

    story_ids = sorted(by_story.keys())
    rng = random.Random(seed)
    chosen = rng.sample(story_ids, min(num_stories, len(story_ids)))

    sampled: list[OpenToMItem] = []
    for sid in chosen:
        sampled.extend(by_story[sid])
    return sampled


def extend_story_sample(
    items: list[OpenToMItem],
    exclude_story_ids: set[str],
    *,
    num_additional: int,
    seed: int = 42,
) -> list[OpenToMItem]:
    """Sample additional stories that are not in ``exclude_story_ids``."""
    by_story: dict[str, list[OpenToMItem]] = {}
    for item in items:
        by_story.setdefault(item.story_id, []).append(item)

    remaining = sorted(sid for sid in by_story if sid not in exclude_story_ids)
    rng = random.Random(seed)
    chosen = rng.sample(remaining, min(num_additional, len(remaining)))

    sampled: list[OpenToMItem] = []
    for sid in chosen:
        sampled.extend(by_story[sid])
    return sampled

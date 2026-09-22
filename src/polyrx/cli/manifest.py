"""Pin dataset revisions and verify downloaded bytes.

``polyrx-manifest pin`` resolves each configured dataset to a concrete Hub
commit, downloads it, and records the sha256 in ``data/MANIFEST.json``. That
file is what lets someone else check they are scoring against the same bytes
the published numbers came from.

``polyrx-manifest verify`` re-hashes what is already on disk and reports
mismatches without downloading anything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from polyrx.cli.compose import load_config
from polyrx.data.fetch import MANIFEST_NAME, DatasetFetcher, Manifest
from polyrx.provenance import file_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polyrx-manifest",
        description="Pin dataset revisions and verify downloaded files.",
    )
    parser.add_argument("action", choices=("pin", "verify", "show"))
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help="Config name under conf/dataset (repeatable). Default: every one.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Hydra overrides, e.g. paths.data_dir=/scratch/data",
    )
    return parser


def _datasets(names: list[str] | None, overrides: list[str]) -> list:
    names = names or ["opentom", "gpqa"]
    return [load_config([f"dataset={name}", *overrides]).dataset for name in names]


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    base = load_config(args.overrides)
    data_dir = base.paths.resolved("data_dir")

    if args.action == "show":
        manifest = Manifest.load(data_dir / MANIFEST_NAME)
        if not manifest.entries:
            print(f"No manifest at {data_dir / MANIFEST_NAME}. Run: polyrx-manifest pin")
            return 1
        for key, entry in sorted(manifest.entries.items()):
            print(f"{key}\n  revision {entry.revision}\n  sha256   {entry.sha256}")
        return 0

    if args.action == "pin":
        fetcher = DatasetFetcher(data_dir, verify=False)
        for dataset in _datasets(args.dataset, args.overrides):
            print(f"Pinning {dataset.name}...")
            fetcher.pin(dataset)
        print(f"Wrote {data_dir / MANIFEST_NAME}")
        return 0

    # verify
    manifest = Manifest.load(data_dir / MANIFEST_NAME)
    if not manifest.entries:
        print(f"No manifest at {data_dir / MANIFEST_NAME}. Run: polyrx-manifest pin")
        return 1
    failures = 0
    for key, entry in sorted(manifest.entries.items()):
        candidates = list(data_dir.rglob(Path(entry.filename).name))
        if not candidates:
            print(f"MISSING  {key} (not downloaded yet)")
            continue
        digest = file_sha256(candidates[0])
        if digest == entry.sha256:
            print(f"ok       {key}")
        else:
            failures += 1
            print(f"MISMATCH {key}\n  expected {entry.sha256}\n  actual   {digest}")
    if failures:
        print(
            f"\n{failures} file(s) differ from the manifest. Results computed from these "
            f"bytes are not comparable to the published numbers."
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

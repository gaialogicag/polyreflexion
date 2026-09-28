"""Check whether a real run would work, before spending a run finding out.

A benchmark run costs money and takes time, and most of the ways it fails are
knowable in advance: a missing key, a model name the provider does not serve, a
gated dataset without a token, Ollama not started. This checks each one and
says what is blocking, rather than failing on the first API call an hour in.

``--live`` sends one minimal request per model role. That is the only check
that can catch a model identifier the provider has retired or renamed, which
no amount of config inspection will tell you.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from polyrx import __version__
from polyrx.cli.compose import load_config
from polyrx.conditions import LOCK_FILENAME, check_against_lock, registry_from_config
from polyrx.config import RootConfig
from polyrx.env import load_env

OK, WARN, FAIL = "ok", "warn", "fail"

_MARK = {OK: "  ok  ", WARN: " warn ", FAIL: " FAIL "}


@dataclass
class Check:
    """One thing that either works or explains itself."""

    name: str
    status: str
    detail: str
    fix: str = ""


def _python_and_package() -> list[Check]:
    version = ".".join(str(n) for n in sys.version_info[:3])
    too_old = sys.version_info < (3, 11)
    checks = [
        Check(
            "python",
            FAIL if too_old else OK,
            f"{version} ({sys.executable})",
            "polyrx needs Python 3.11 or newer." if too_old else "",
        ),
        Check("polyrx", OK, f"{__version__}"),
    ]
    for module, extra, what in (
        ("matplotlib", "viz", "charts and topology renderings"),
        ("pyarrow", "gpqa", "GPQA domain labels"),
    ):
        try:
            __import__(module)
            checks.append(Check(module, OK, f"present — {what} available"))
        except ImportError:
            checks.append(
                Check(
                    module,
                    WARN,
                    f"missing — {what} will be skipped",
                    f'pip install -e ".[{extra}]"',
                )
            )
    return checks


def _config(cfg: RootConfig) -> list[Check]:
    checks: list[Check] = []
    conditions = registry_from_config(cfg.conditions.conditions)
    checks.append(Check("conditions", OK, f"{len(conditions.names())} in the grid"))

    lock = cfg.paths.resolved("data_dir") / LOCK_FILENAME
    if not lock.is_file():
        checks.append(Check("condition lock", WARN, "no lock recorded", "polyrx-conditions lock"))
    else:
        problems = check_against_lock(conditions, lock)
        checks.append(
            Check(
                "condition lock",
                FAIL if problems else OK,
                f"{len(problems)} mismatch(es)" if problems else "grid matches the lock",
                "polyrx-conditions lock, deliberately" if problems else "",
            )
        )

    prompts = cfg.prompts
    checks.append(
        Check(
            "prompts",
            OK if prompts.reflexion.qa else WARN,
            "qa template present"
            if prompts.reflexion.qa
            else "no qa template — this set cannot score a benchmark",
            "" if prompts.reflexion.qa else "prompts/reflexion=default",
        )
    )
    return checks


def _credentials(cfg: RootConfig, live: bool) -> list[Check]:
    checks: list[Check] = []
    from polyrx.config import HostedModelConfig

    # Any role can be served by any provider, so which ones need a key is read
    # off the config rather than assumed. A role on a local runtime has no key
    # to check and is covered by the local-backend section instead.
    roles = {
        name: getattr(cfg.backends, name)
        for name in ("answerer", "judge", "meta_judge", "open_weights")
    }

    for role, backend in roles.items():
        if not isinstance(backend, HostedModelConfig):
            continue
        key = os.environ.get(backend.api_key_env, "")
        if not key:
            checks.append(
                Check(
                    f"{role} key",
                    FAIL,
                    f"{backend.api_key_env} is not set",
                    f"export {backend.api_key_env}=... or put it in .env",
                )
            )
            continue
        provider = type(backend).__name__.replace("Config", "").lower()
        checks.append(
            Check(
                f"{role} key",
                OK,
                f"{provider}: {backend.api_key_env} set ({len(key)} chars)",
            )
        )

        if not live:
            checks.append(
                Check(
                    f"{role} model", WARN, f"{backend.model} — not verified", "re-run with --live"
                )
            )
            continue

        checks.append(_probe_model(role, backend))
    return checks


#: Output allowance for a probe. A reasoning model bills its thinking against
#: this budget, so a probe of 1 token fails on every such model with a 400 that
#: reads like a broken model name. Large enough to let thinking start, small
#: enough that a full doctor run costs a fraction of a cent.
_PROBE_MAX_TOKENS = 64


def _probe_model(role: str, backend) -> Check:
    """One minimal completion, to prove the model identifier is served."""
    from polyrx.config import GeminiConfig

    try:
        if isinstance(backend, GeminiConfig):
            _probe_gemini(backend)
        else:
            _probe_openai(backend)
        return Check(f"{role} model", OK, f"{backend.model} answered")
    except Exception as exc:
        text = str(exc)
        hint = ""
        lowered = text.lower()
        if "does not exist" in lowered or "model_not_found" in lowered:
            hint = f"backends.{role}.model names a model this account cannot use."
        elif "authentication" in lowered or "api key" in lowered or "401" in text:
            hint = f"The key in {backend.api_key_env} was rejected."
        elif "rate" in lowered or "429" in text:
            hint = "Rate limited. The model exists; try again shortly."
        elif "google" in lowered and "genai" in lowered:
            hint = 'The Gemini backend needs: pip install -e ".[gemini]"'
        return Check(f"{role} model", FAIL, f"{backend.model}: {text[:160]}", hint)


def _probe_openai(backend) -> None:
    from polyrx.models.openai_client import OpenAIClient

    client = OpenAIClient(backend)
    client._client.chat.completions.create(
        model=backend.model,
        messages=[{"role": "user", "content": "ok"}],
        max_completion_tokens=_PROBE_MAX_TOKENS,
    )


def _probe_gemini(backend) -> None:
    from polyrx.models.gemini_client import GeminiClient

    client = GeminiClient(backend)
    client._client.models.generate_content(
        model=backend.model,
        contents="ok",
        config=client._request_config,
    )


def _local_backend(cfg: RootConfig) -> list[Check]:
    from polyrx.config import OllamaConfig

    ollama = cfg.backends.open_weights
    # Only the Ollama runtime has a daemon to find and a model to pull. Served
    # any other way, the role is an ordinary API backend and was already checked
    # with the rest of them.
    if not isinstance(ollama, OllamaConfig):
        provider = type(ollama).__name__.replace("Config", "").lower()
        return [Check("open_weights", OK, f"served by {provider}, not a local runtime")]
    if shutil.which("ollama") is None:
        return [
            Check(
                "ollama",
                WARN,
                "not installed — open_weights_* conditions cannot run",
                "https://ollama.com, then: ollama pull " + ollama.model,
            )
        ]
    try:
        import json
        import urllib.request

        with urllib.request.urlopen(f"{ollama.host.rstrip('/')}/api/tags", timeout=3) as resp:
            names = [m["name"] for m in json.load(resp).get("models", [])]
    except Exception:
        return [Check("ollama", WARN, f"not reachable at {ollama.host}", "ollama serve")]

    have = any(n == ollama.model or n.startswith(f"{ollama.model}:") for n in names)
    return [
        Check(
            "ollama",
            OK if have else WARN,
            f"running; {ollama.model} {'pulled' if have else 'NOT pulled'}",
            "" if have else f"ollama pull {ollama.model}",
        )
    ]


def _dataset(cfg: RootConfig, live: bool) -> list[Check]:
    from polyrx.data.fetch import MANIFEST_NAME, Manifest
    from polyrx.datasets import get_adapter

    dataset = cfg.dataset
    checks: list[Check] = []

    try:
        get_adapter(dataset)
        checks.append(Check("dataset adapter", OK, f"{dataset.name} -> {dataset.adapter}"))
    except ValueError as exc:
        return [Check("dataset adapter", FAIL, str(exc)[:200])]

    source = dataset.primary
    if source.path:
        path = Path(source.path).expanduser()
        checks.append(
            Check(
                "dataset source",
                OK if path.is_file() else FAIL,
                f"local file {path}" + ("" if path.is_file() else " — not found"),
            )
        )
    else:
        gated_note = " (gated)" if source.gated else ""
        checks.append(
            Check("dataset source", OK, f"{source.repo_id}/{source.filename}{gated_note}")
        )
        if source.revision in {"", "main", "master", "HEAD"}:
            checks.append(
                Check(
                    "dataset pin",
                    WARN,
                    f"revision is {source.revision or 'unset'!r} — upstream can change it",
                    "polyrx-manifest pin",
                )
            )
        if source.gated and not os.environ.get("HF_TOKEN"):
            checks.append(
                Check(
                    "HF_TOKEN",
                    WARN,
                    "unset — the gated source will fall through to a fallback",
                    "Accept the dataset terms on the Hub, then export HF_TOKEN=...",
                )
            )

    manifest = Manifest.load(cfg.paths.resolved("data_dir") / MANIFEST_NAME)
    checks.append(
        Check(
            "dataset manifest",
            OK if manifest.entries else WARN,
            f"{len(manifest.entries)} file(s) recorded" if manifest.entries else "no manifest",
            "" if manifest.entries else "polyrx-manifest pin",
        )
    )

    if live:
        checks.extend(_probe_dataset(cfg))
    return checks


def _probe_dataset(cfg: RootConfig) -> list[Check]:
    """Fetch the dataset and build its items, which is the only real proof.

    Fetching alone proves access and nothing else. Parsing the file into items
    is what catches an adapter reading the source wrongly, and that is worth
    knowing here rather than after an hour of answering.
    """
    from polyrx.data.fetch import DatasetFetcher

    try:
        fetcher = DatasetFetcher(
            cfg.paths.resolved("data_dir"), verify=cfg.dataset.verify_checksums
        )
        path = fetcher.fetch_dataset(cfg.dataset)
        size = path.stat().st_size
    except Exception as exc:
        return [Check("dataset fetch", FAIL, str(exc)[:200])]

    fetched = Check("dataset fetch", OK, f"{path.name} ({size / 1024:.0f} KiB)")
    return [fetched, _probe_items(cfg, path)]


def _probe_items(cfg: RootConfig, path: Path) -> Check:
    """Parse the source into items and check they can be answered at all."""
    from polyrx.datasets import get_adapter

    try:
        adapter = get_adapter(cfg.dataset)
        items = adapter.load(path)
    except Exception as exc:
        return Check("dataset items", FAIL, f"{type(exc).__name__}: {str(exc)[:160]}")

    if not items:
        return Check("dataset items", FAIL, "the adapter produced no items")

    # `load_items` raises above the threshold and prints below it. Doctor
    # reports rather than raises, so the same rule is applied to the same
    # numbers here instead of calling it.
    unanswerable = [item for item in items if not item.gold_is_allowed()]
    fraction = len(unanswerable) / len(items)
    if not unanswerable:
        return Check("dataset items", OK, f"{len(items)} items, all answerable")

    detail = (
        f"{len(unanswerable)} of {len(items)} ({fraction:.1%}) have a gold label outside "
        f"their own label space"
    )
    example = unanswerable[0]
    fix = (
        f"e.g. {example.item_id}: gold {example.gold_label!r} not in "
        f"[{example.label_space}]. Check how the {cfg.dataset.adapter!r} adapter "
        f"derives label_space."
    )
    over = fraction > cfg.dataset.max_unanswerable_fraction
    if over:
        # The same threshold a real run applies, reported here instead of raised.
        detail += (
            f" — above dataset.max_unanswerable_fraction "
            f"({cfg.dataset.max_unanswerable_fraction:.1%})"
        )
    return Check("dataset items", FAIL if over else WARN, detail, fix)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="polyrx-doctor",
        description="Check whether a real benchmark run would work.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Send one minimal request per model and fetch the dataset. "
        "The only way to catch a retired model name or a gated repo.",
    )
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. dataset=gpqa")
    args = parser.parse_args(argv)

    # Before any check reads os.environ. A key in .env is a configured key, and
    # reporting it missing sends the user to fix something that is not broken.
    load_env()

    try:
        cfg = load_config(args.overrides)
    except SystemExit:
        raise
    except Exception as exc:
        print(f" FAIL  config did not compose: {exc}")
        return 1

    groups = [
        ("Environment", _python_and_package()),
        ("Configuration", _config(cfg)),
        ("Model access", _credentials(cfg, args.live)),
        ("Local backend", _local_backend(cfg)),
        (f"Dataset: {cfg.dataset.name}", _dataset(cfg, args.live)),
    ]

    fixes: list[str] = []
    worst = OK
    for title, checks in groups:
        print(f"\n{title}")
        for check in checks:
            print(f"  [{_MARK[check.status]}] {check.name:18s} {check.detail}")
            if check.fix:
                fixes.append(f"{check.name}: {check.fix}")
            if check.status == FAIL or (check.status == WARN and worst == OK):
                worst = check.status if check.status == FAIL else WARN

    print()
    if worst == FAIL:
        print("A real run would fail. Blocking problems above.")
    elif worst == WARN:
        print("A real run would work, with the caveats above.")
    else:
        print("Ready for a real run.")

    if fixes:
        print("\nTo fix:")
        for fix in dict.fromkeys(fixes):
            print(f"  - {fix}")
    if not args.live:
        print("\nModel names and dataset access are unverified. Re-run with --live to check them.")
    return 1 if worst == FAIL else 0


if __name__ == "__main__":
    sys.exit(main())

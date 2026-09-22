# Contributing

## Setup

```bash
conda env create -f environment.yml && conda activate polyrx
# or: python3.12 -m venv .venv && source .venv/bin/activate
#     pip install -e ".[dev,viz,gpqa]"
cp .env.example .env              # put your OPENAI_API_KEY here
polyrx-doctor                     # what is still missing
```

Check the install with the offline path, which needs no key and makes no
network calls:

```bash
polyrx-bench experiment=smoke
```

## Design principles

These are enforced in review, and mirrored in `.cursor/rules/` for Cursor users.

1. **Configuration lives in `conf/` at the repository root, not in code.** No
   `os.environ.get` outside `polyrx/config.py`. The one exception is
   reading an API key by the variable name the config gives.
   An experiment config carries `# @package _global_` and is composed last, so
   it can override any group; a group config composes onto its schema
   (`base_backends`, `base_dataset`, …) so a typo fails at composition.
2. **Derive, do not tabulate.** Anything computable from a condition's backend,
   depth and meta budget is a property on `Condition`, not another dictionary.
3. **The meta layer selects; the engine produces.** Nothing in `meta/` writes
   reasoning text. If a change makes a judge generate an answer, it belongs in
   the engine instead.
4. **Recursion where it fits**, with explicit base cases.
5. **Parallel by default** for independent model calls and subtrees.
6. **Comment the why.** Assume a reader who knows Python but not this project.
7. **Exercise a change offline before proposing it.** `experiment=smoke` runs
   the whole pipeline on stub clients, so a broken call site shows up without
   spending a token.

## Before opening a pull request

```bash
ruff check src && ruff format --check src
mypy
polyrx-bench experiment=smoke
```

## Changing a prompt

Templates live in `conf/prompts/`, not in the package. Edit the YAML; there is
no Python to touch. A new set is a new file in `conf/prompts/reflexion/` or
`conf/prompts/meta/`, composed onto its schema (`base_reflexion_prompts` /
`base_meta_prompts`) so a misspelled template name fails at composition.

Prompt edits invalidate cached summaries. Bump `experiment.cache_namespace`
when you change a template, otherwise the next run silently reuses summaries
built by the old prompt and the number you publish corresponds to no version of
the code.

## Adding a condition

Add one entry to `conf/conditions/published.yaml`. Cache key, prior budget,
summary behaviour and report ordering follow from it. Do not add a lookup
table, and do not put the grid back into Python.

Then re-lock and say in the pull request which published results the change
invalidates:

```bash
polyrx-conditions check
polyrx-conditions lock
```

## Adding a dataset

A config file in `conf/dataset/`, usually with `adapter: tabular` and a column
mapping. No Python. See `conf/dataset/example_mcq.yaml`.

Write an adapter only when the source needs parsing the column mapping cannot
express — allowed answers that vary per item, or two incompatible source
layouts. Subclass `DatasetAdapter`, implement `load` and `match_label`, and
decorate the class with `@register_adapter`.

Nothing outside `polyrx/datasets/` may branch on which dataset is running.

## Changing a dataset revision

```bash
polyrx-manifest pin --dataset gpqa
polyrx-manifest verify
```

Commit the updated `data/MANIFEST.json` in the same pull request as the config
change, and say in the description which published numbers it invalidates.

## Commits

Conventional Commits: `feat|fix|docs|style|refactor|test|chore(scope): description`,
imperative mood, lowercase, no trailing period, subject under 72 characters.
Branches are `<type>/<short-description>`. Never commit to `main` directly.

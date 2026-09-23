# polyreflexion

Recursive text reflexion over a Sierpinski triangle of three perspectives,
steered by a polycontextural meta layer, with the OpenToM and GPQA Diamond
evaluation suites used to measure it.

The meta layer never writes reasoning text. It **selects**; the engine
**produces**.

## What it does

The engine takes an input text and recursively splits it across three
perspectives — objectivity (**O**), subjectivity (**S**), conceptuality (**B**)
— producing a tree whose leaves are re-integrated into a single summary.
Independent subtrees run concurrently.

The meta layer then observes a completed engine run and decides what the next
recursive cycle should be:

| Stage | What it does |
|---|---|
| `PolyJudge` | Three judges score the answer in parallel from the subjective (A/F), objective (W/F) and dialectical (W/R) contexts |
| `Interpreter` | Deterministic lookup: 3 positives → `W` converged, 2 → `R` expand, 1 → `A` refine, 0 → `F` reset |
| `GeometryMapper` | Maps each value to a canonical Sierpinski move plus the preserved unused alternative (the *semantic surplus*) |
| `StrategyPlanner` | Expansion negates the positive verdicts into new boundary statements; refinement reuses the negative rationales as sub-triangle coordinates; `F` resets once at higher depth |
| `MetaController` | Bounded loop with oscillation, semantic-drift, low-confidence and engine-failure guards |

On meta cycle 0 only, a fully positive `(A,W,W)` is read as `(A,W,R)` so the
loop expands once instead of terminating before it has done anything. From
cycle 1 onward `(A,W,W)` terminates normally.

## Install

Python 3.11 or newer. With conda:

```bash
conda env create -f environment.yml
conda activate polyrx
```

Or with a plain virtual environment:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,gpqa]"
```

Then set up credentials and check what is missing:

```bash
cp .env.example .env        # put OPENAI_API_KEY here, GEMINI_API_KEY for Gemini
polyrx-doctor               # says exactly what would block a real run
```

`polyrx-doctor` checks the interpreter, the config, the condition lock, the
keys, the local backend and the dataset. Add `--live` and it sends one minimal
request per model and fetches the dataset — the only way to catch a model name
the provider has retired, or a gated repository you have no token for.

Everything below works offline with `experiment.use_stub=true`; only real runs
need a key.

Extras: `viz` adds chart and topology rendering (matplotlib); `gpqa` adds the
parquet reader used to recover GPQA domain labels; `gemini` adds the Google SDK
the Gemini backends need. All three are genuinely optional — without `viz`,
reports and traces are still written and the figures are skipped with a note.

For a local open-weights backend, install [Ollama](https://ollama.com) and pull
a model: `ollama pull phi4-mini-reasoning`. Serving with
`OLLAMA_NUM_PARALLEL=4` is worth it — the engine issues independent calls
concurrently.

## Run it

Everything is configured through [Hydra](https://hydra.cc). The config tree is
`conf/` at the repository root — read it, copy it, edit it. Nothing under
`src/` is configuration: **the prompt templates live in `conf/prompts/` too**,
because a prompt is the thing a researcher changes most and burying it in the
package made it the hardest thing to change.

Commands are run from a checkout. The `conf/` directory is located by searching
upward from the working directory, so running from a subdirectory works too.
Point `POLYRX_CONF` at a tree somewhere else to override the search.

```bash
# Offline smoke test: no API key, no network, deterministic stub responses.
polyrx-bench experiment=smoke

# One question, baseline reflexion beside meta reflexion.
polyrx-ask "Why do people keep secrets?"

# One meta-guided run with a full trace and Sierpinski topology renderings.
polyrx-reflect --max-cycles 3 "Your question"
```

Benchmarks:

```bash
# OpenToM (theory of mind)
polyrx-bench experiment=opentom experiment.num_items=20 \
    experiment.cache_namespace=ot_v1 experiment.detailed=true

# GPQA Diamond (graduate biology / physics / chemistry)
polyrx-bench experiment=gpqa experiment.num_items=100 \
    experiment.cache_namespace=gpqa_v1 experiment.detailed=true

# Anything else you have configured
polyrx-bench dataset=my_dataset experiment.num_items=50
```

Override any setting on the command line, and sweep with `-m`:

```bash
# Swap the engine model without touching a call site.
polyrx-bench experiment=gpqa backends.nano.model=gpt-4o-mini

# Incremental meta budgets. Each run merges into the previous one's results,
# and each budget continues from the previous budget's cached summaries.
polyrx-bench -m experiment=gpqa experiment.merge_from=latest \
    'experiment.conditions=[nano_meta_c1],[nano_meta_c2],[nano_meta_c3]'
```

`merge_from` adds conditions to the **same** items. `extend_from` adds new
items under the same conditions. `merge_from=latest` resolves to the newest run
file of that suite, so a sweep does not need a pasted timestamp.

## Prompts

Templates are config groups, not files inside the package:

```
conf/prompts/reflexion/{default,gpqa,ask}.yaml    # engine templates
conf/prompts/meta/{default,opentom,gpqa,ask}.yaml # judge / boundary templates
```

```bash
polyrx-bench prompts/meta=gpqa                              # select a set
polyrx-bench -m prompts/meta=default,opentom,gpqa           # sweep sets
polyrx-bench 'prompts.meta.drift_check="..."'               # override one template
```

`{name}` placeholders are `str.format` fields and a literal brace is `{{`.
Hydra's override grammar reserves `{`, so a command-line value containing a
placeholder needs inner quotes, as above.

An experiment selects its sets like any other group — `experiment=gpqa` pulls
`prompts/reflexion=gpqa` and `prompts/meta=gpqa`. Each template's sha256 is
recorded individually in the run's provenance block, so comparing two runs
names the template that changed rather than only saying the set differs.

Changing a template invalidates cached summaries: bump
`experiment.cache_namespace` when you edit one.

## Other settings

Every tunable is a config group; nothing that changes behaviour is a literal in
the source.

| Group | What it holds |
|---|---|
| `postprocess` | When a generation counts as collapsed and gets retried, and how a final answer is pulled out of reasoning text. **These change scores.** `postprocess=lenient` loosens them |
| `report` | Chart sizes, table truncation, row caps, progress interval. Presentation only |
| `provenance` | Which package versions are recorded, git subprocess timeout, hash chunk size |
| `engine` | Worker pool size and cap, default tree depth |
| `dataset` | Hub source and revision, plus the hash lengths that form item ids |

```bash
polyrx-bench postprocess=lenient                     # accept more generations
polyrx-bench report.progress_every=50                # quieter output
polyrx-bench postprocess.degeneracy.max_digit_ratio=0.3
```

Two caveats. `postprocess` decides which summaries get a second attempt, so
changing it changes published numbers — bump `experiment.cache_namespace` with
it. And `dataset.id_hash_chars` is a data format rather than a knob: changing
it renames every item, invalidating every cached summary and making saved runs
unmergeable.

A few numbers are deliberately *not* configurable, because they are structure
rather than settings: the three-way branching of the Sierpinski tree, the
brace-depth counter in the `\boxed{}` parser, console rule widths. Exposing
those would let a value silently break the algorithm.

## Conditions

A condition is one cell of the experiment grid: a backend, a reflexion depth,
and a meta cycle budget. Everything else — cache key, whether a summary pass is
needed, which budget it continues from — is derived from those three numbers.

The grid is config, in `conf/conditions/`. `polyrx/conditions.py` defines only
what a condition *is*.

```bash
polyrx-conditions show          # print the grid with its derived cache keys
polyrx-bench conditions=nano_only
```

| Name | Backend | Depth | Meta cycles |
|---|---|---|---|
| `nano_direct` | nano | 0 | 0 |
| `nano_reflexion` | nano | 1 | 0 |
| `nano_reflexion_d2` | nano | 2 | 0 |
| `nano_reflexion_d3` | nano | 3 | 0 |
| `nano_meta_c1` … `nano_meta_c4` | nano | 1 | 1 … 4 |
| `phi_direct` | phi | 0 | 0 |
| `phi_reflexion` | phi | 1 | 0 |
| `phi_reflexion_d2` | phi | 2 | 0 |

Group shorthands work on the command line: `nano` is every nano condition,
`nano_nod3` is the same without the expensive depth-3 run, `all` is everything.

Adding a condition is one entry in `conf/conditions/published.yaml`. There is
no lookup table to keep in sync.

`data/conditions.lock.json` records the grid the published results were
produced with, the same way `data/MANIFEST.json` records dataset revisions. A
run compares the two and stops when they disagree, because every cached summary
is keyed by a condition's `summary_key`: change one and the cache is orphaned
silently rather than failing.

```bash
polyrx-conditions check         # compare the grid against the lock
polyrx-conditions lock          # re-lock, deliberately
```

## Backends

Call sites ask for a **role**, never a model name:

| Role | Used for | Default |
|---|---|---|
| `nano` | Reflexion engine and meta cycles | `gpt-5.4-nano` |
| `judge` | Benchmark label scoring | `gpt-5-mini` |
| `meta_judge` | The three polycontextural judges | `gpt-4o` |
| `phi` | Local open-weights backend via Ollama | `phi4-mini-reasoning` |

`meta_judge` is deliberately a different model from `judge` so meta evaluation
stays independent of benchmark scoring.

Change the models in `conf/backends/openai.yaml`, or pick a different file:

```bash
polyrx-bench backends=gemini     # every hosted role on Google Gemini
polyrx-bench backends=local      # every role through Ollama
```

Each hosted role names its provider in the backends file's `defaults`, so the
provider is a per-role choice, not a per-run one. Answering with Gemini while
keeping the published OpenAI judges:

```bash
polyrx-bench backends/role@backends.nano=gemini \
  backends.nano.model=gemini-2.5-flash \
  backends.nano.api_key_env=GEMINI_API_KEY
```

Gemini goes through the `google-genai` SDK rather than an OpenAI-compatible
shim, because the shim drops `thinking_budget` — and on these models how much
the model is allowed to think changes the answer, so it belongs in the recorded
config. It needs `pip install -e ".[gemini]"` and `GEMINI_API_KEY`.

| Provider | Config class | Key variable | Extra fields |
|---|---|---|---|
| OpenAI and compatible gateways | `OpenAIConfig` | `OPENAI_API_KEY` | `base_url` |
| Google Gemini | `GeminiConfig` | `GEMINI_API_KEY` | `thinking_budget`, `max_output_tokens` |
| Ollama, local | `OllamaConfig` | none | `host`, `num_ctx`, `num_predict` |

Only API keys come from the environment. A config object holds the *name* of
the variable to read, never the value, so a resolved config can be written into
a results file without redacting anything.

## Datasets

Nothing in the code knows which benchmark it is running. A dataset is a config
file naming an adapter and a source:

```bash
polyrx-bench dataset=gpqa
polyrx-bench dataset=my_dataset          # your own conf/dataset/my_dataset.yaml
```

**Adding one usually needs no Python.** The `tabular` adapter reads CSV, TSV,
JSON and JSONL, from the Hub or from a local path, and is driven entirely by a
column mapping. Copy `conf/dataset/example_mcq.yaml`:

```yaml
name: my_dataset
adapter: tabular
group_name: subject        # what to break results down by
sample_by: item            # `context` when several items share a passage

fields:
  context: passage         # optional: the text to reason over
  question: q
  answer: a
  label_space: allowed     # or omit, and give a fixed label_space
  group: subject
  # multiple choice, either layout:
  # choices: [option_a, option_b, option_c, option_d]
  # correct_choice: "Correct Answer"
  # incorrect_choices: ["Wrong 1", "Wrong 2", "Wrong 3"]

primary:
  path: /data/my_dataset.csv     # or repo_id + filename + revision
```

Write an adapter only when the source needs real parsing. Both shipped ones do,
for reasons worth knowing: OpenToM's allowed answers depend on the question,
and GPQA's official release and public mirror have different shapes. Subclass
`DatasetAdapter`, implement `load` and `match_label`, decorate with
`@register_adapter`, and name it in your dataset config.

Neither shipped dataset is redistributed here. Both are fetched from the
Hugging Face Hub at a pinned revision and verified against `data/MANIFEST.json`.

```bash
polyrx-manifest pin       # resolve each dataset to a commit, record its sha256
polyrx-manifest verify    # re-hash what is on disk, report any mismatch
polyrx-manifest show      # print the current pins
```

A run whose dataset is pinned to `main`, or has no recorded checksum, says so
in its output — upstream can change those bytes under you, and then a published
number no longer corresponds to any data.

GPQA's official release (`Idavidrein/gpqa`) is gated: accept the terms on the
Hub and set `HF_TOKEN`. Without it the loader falls through to the public
mirror, which has no domain labels; those are then recovered from a third
metadata repository.

## Reproducibility

Every run writes a `provenance` block beside its results: the git commit and
whether the tree was dirty, the package version, the resolved model identifiers
actually called, dataset revisions and checksums, and the sha256 of every
prompt template loaded. Reasons a run is *not* exactly reproducible — a dirty
tree, an unpinned dataset, a missing checksum — are printed at the end of the
run rather than left for a reader to notice.

One thing worth knowing when comparing a meta condition against plain
reflexion: meta cycle 0 runs its own depth-1 reflexion, independent of the
`nano_reflexion` summary cache. Part of any gap between `nano_meta_cN` and
`nano_reflexion` is therefore sampling variance rather than the meta layer.

Prompt edits invalidate cached summaries. Bump `experiment.cache_namespace`
whenever a template changes, or the next run silently reuses summaries built by
the old prompt.

## Layout

| Path | Role |
|---|---|
| `conf/` | Hydra config tree, including all prompt templates (not packaged) |
| `polyrx/engine.py` | `ReflexionEngine` — the recursive tree and its parallel pools |
| `polyrx/conditions.py` | The experiment grid as data |
| `polyrx/config.py` | Every configurable knob, as typed dataclasses |
| `polyrx/meta/` | Judges, interpretation, geometry, strategy, controller, trace, rendering |
| `polyrx/config.py` (prompt dataclasses) | Schema the prompt YAML is checked against |
| `polyrx/models/` | LLM clients and the role registry |
| `polyrx/datasets/` | The dataset interface, the config-driven tabular adapter, and one adapter per awkward source |
| `polyrx/benchmark/` | Runner, judge, metrics and reports — one of each, dataset-neutral |
| `polyrx/data/` | Pinned, checksum-verified dataset fetching |
| `polyrx/provenance.py` | What a reader needs to reproduce a run |
| `polyrx/charts.py` | Optional matplotlib access; charts are skipped, never fatal |
| `polyrx/cli/` | Console entry points |

## Citing

See `CITATION.cff`.

## License

MIT. See `LICENSE`.

# driftwatch

Label-free detection of silent quality degradation in deployed language models.

A monitor that detects quality degradation from the statistical shape of
generated text alone - no labels at detection time - plus the evaluation harness
that proves it works. Four faults are injected at known onsets; the monitor is
calibrated on healthy data only, frozen, then scored on detection latency and
false-alarm rate.

## Status

**Implemented.** Every function is written to the contract in its docstring,
and the full suite passes offline - no network, no credentials:

```bash
uv sync
uv run pytest -m gate    # 61 architectural gates - must always pass
uv run pytest -m spec    # behavioural specs, incl. the Stage 0-7 gates
```

The four freeze tests require a **clean git tree**: a freeze stamps the current
commit and refuses to point at code that was never committed. Commit, then run.

Generation runs **locally, on your own machine, through Ollama**. The batch
platform path (`expand` / `submit` / `resume`) is kept and tested, but it is no
longer the default. Before a real round, every stage is exercised against
synthetic data and in-memory fakes of both backends
(`tests/fake_ollama.py`, `tests/fake_backend.py`).

## Running locally

**One-time setup.** Install Ollama, start the server, and pull the five pinned
builds (about 37 GB):

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=8192 ollama serve
```

```bash
for m in gemma4:26b-a4b-it-q4_K_M gemma4:e4b-it-q4_K_M gemma4:e2b-it-q4_K_M gemma3:12b-it-q4_K_M nomic-embed-text:v1.5; do ollama pull $m; done
```

- `OLLAMA_NUM_PARALLEL` is how many requests the server runs at once; pass the
  same number to `--concurrency`.
- `OLLAMA_CONTEXT_LENGTH=8192` matters: the longest prompts (XSum articles,
  banking77 with its 77 labels) plus a 512-token answer must fit, and Ollama
  TRUNCATES an oversized prompt silently rather than failing.
- `.env` points at the local server:
  `DRIFTWATCH_BASE_URL=http://localhost:11434/v1`, `DRIFTWATCH_API_KEY=ollama`
  (Ollama ignores the key, but the setting is required).

**The run.** Keep the machine on power and awake (`caffeinate -dims` on macOS).

```bash
uv run driftwatch corpus build
uv run driftwatch probe                              # STOP if this fails - see below
scripts/generate_all.sh                              # every round and its embeddings, in run order
uv run driftwatch calibrate --completions data/completions.parquet --without refusal_rate --without format_validity
git commit ...                                       # freeze refuses a dirty tree
uv run driftwatch freeze --thresholds data/thresholds.proposed.json --completions data/completions.parquet
scripts/analyse_all.sh                               # every evaluation and stress test, v1-v3
```

The rounds, as run: r001 healthy, r002 F3, r003 F2, r004 F1, r005 F5, then
r006 F7 (the healthy model on fresh seeds, a negative control). F8, a
system-prompt fault built to keep answer length, was designed but not
generated: neither candidate prompt passed the pre-registered pilot
(`driftwatch pilot-f8`, rule and result in `config/extension_prompts.yaml`). F4 and F6 have no round: both are derived
from the healthy completions.

v2 and v3 were calibrated after v1 with `calibrate --scoring per_slice` /
`per_slice_capped` and frozen to their own files with `freeze --out`; the
original record was never touched. `scripts/analyse_all.sh` lists every
analysis command with its arguments: `evaluate`, `report`, `gradual`,
`mixtures`, `ablation`, `sensitivity`, `length-matched`, `one-slice`,
`sequential`, `half-sample`, `one-answer`, `jsonx-check` and `leakage-check`. Every one that replays verifies
its frozen record first and refuses on any mismatch.

`data/logs/` keeps the logs of the real run, with the one-off shell scripts
that drove it as they were at the time (`chain.sh`, `chain_f5.sh`,
`after_f5.sh`, `phase11.sh`, `chain_f7.sh`). They are a record, not tools:
`scripts/` holds the reusable versions.

**Interrupted? Run the same command again.** `generate` and `embed` append
every answer to `data/rounds/<round_id>/results.jsonl` the moment it arrives;
a rerun sends only what is missing. A crashed or overloaded server is never
recorded as a model failure - those rows are simply retried. When every row is
in, the round is reconciled (C10) and stored once.

**The model on disk must be the pinned build.** `generate` compares the digest
Ollama reports with the one in `config/models.yaml` and refuses to run on a
mismatch. Every row records the real digest (C9), and
`data/rounds/<round_id>/round.json` records the Ollama version and settings.

**All rounds must come from one machine and one set of builds.** Thresholds are
calibrated against specific model digests; mixing rounds from two machines or
two builds silently invalidates them.

`driftwatch evaluate --synthetic` runs Stages 5-7 end to end on synthetic data,
without a freeze.

### The batch platform (optional)

The Sheshnag path still works: `expand`, `submit --chunks N`, `resume`, and
`probe --backend batch` / `embed --backend batch`. `models.yaml` would then
need that deployment's catalogue ids instead of Ollama tags.

## Workstreams

Freeze `settings.py`, `corpus/schemas.py` and `signals/base.py` first - all four
tracks import them. Then:

| Track | Owns | Blocked by |
|---|---|---|
| T1 | `corpus/`, `synthetic/` | schemas |
| T2 | `provider/` | schemas, settings |
| T3 | `signals/` | synthetic fixtures |
| T4 | `detect/`, `evaluate/`, `oracle/` | signals protocol |

**T1's `synthetic/generate.py` is the critical path.** T3 and T4 both develop
against its output, so nothing else can start in earnest until it works.

## Build order

Each stage has a gate; do not proceed past a failing one.

0. Skeleton + synthetic generator - round-trips the parquet schema
1. Corpus - 5,000 rows, exact split counts, reproducible from seed
2. Provider probe - **seed determinism** (see below)
3. Expand/submit/collect - reconciliation fails loudly on a dropped row
4. Signals - flat at knob 0.0, monotone in the targeted knob
5. Calibrate + freeze - leakage raises; heldout false-alarm rate within 2x
6. Oracle - `detect/` still cannot reach `oracle/`
7. Evaluate - severe fault detected fast, zero-magnitude fault not detected
8. Reporting - latency as a distribution, per fault, never pooled

## Before spending a real round

```bash
uv run driftwatch probe --rows 20
```

Sends one prompt x 10 distinct seeds and x 10 identical seeds to the healthy
model. Run it at the `--concurrency` you will generate with: when Ollama
batches parallel requests together, identical seeds can stop reproducing, and
this is where that shows up. If it fails only at high concurrency, generate at
a lower one.

**If distinct seeds do not vary the output, stop.** S3 (self-consistency) is not
viable and the design must change before a baseline round is spent.

## Constraints

These are not style preferences. Violating any of them silently invalidates the
experiment; the ones marked *enforced* fail the build.

| # | Constraint | |
|---|---|---|
| C1 | Never request `n > 1` - k samples means k rows | *enforced* |
| C2 | Distinct seed per sample of a query | *enforced* |
| C3 | `detect/` never imports `oracle/` | *enforced* |
| C4 | Thresholds calibrated on `calib` only, then frozen | *enforced* |
| C5 | `heldout` used only for false-alarm measurement | *enforced* |
| C6 | Unparseable completions are counted, never dropped | |
| C7 | No in-process model inference - models run in a separate server (Ollama), pinned by digest | *enforced* |
| C8 | F1-F3 share one query subset; only F4 changes queries | |
| C9 | Log `model_id`, `digest` and `seed` on every row | |
| C10 | Reconcile returned against submitted rows every round | |

Three that are easy to break by accident:

- **`n > 1` is not caught at submission.** The platform's validator never
  inspects it. The row passes, reaches a worker, and fails *there* - so a single
  mistake costs a full round and surfaces only as scattered failed rows.
- **The JSONL row schema is strict.** Only `custom_id`, `method`, `url`, `body`
  are allowed and anything else rejects the whole file, which is why row
  metadata is encoded into `custom_id`.
- **`jsonx` rows must not request strict `json_schema` mode.** The platform
  would validate server-side and fail the row, so malformed JSON would never
  come back as text and S5 would silently read zero forever.

## Layout

Modules live directly under `src/`, which is mapped to the install root. The
package is installed rather than imported from the working directory, so a
broken install fails loudly instead of silently resolving against loose source
files.

```
src/settings.py   configuration + all corpus/window arithmetic
src/cli.py        typer app
src/log.py        structured logging  (NOT logging.py — that shadows the stdlib)
src/textnorm.py   answer normalisation shared by S3 and the oracle (imports neither)
src/config/       models.yaml, prompts.yaml, refusal_patterns.yaml, jsonx_schema.json,
                  banking77_labels.yaml
src/corpus/       query construction, splits, leakage check  (code; data lives in data/)
src/provider/     expand, collect; local (Ollama, the default); batch client, submit, manifest
src/signals/      windows, groups (shared scoring base) + S1..S7
src/detect/       calibrate, detectors, freeze
src/oracle/       ground-truth scoring (never imported by detect/)
src/evaluate/     replay, metrics, bootstrap, pipeline (the protocol, step by step);
                  gradual, ablation, mixtures, sensitivity, lengthmatched, oneslice,
                  sequential, halfsample, single (the stress tests)
src/synthetic/    offline completion generator
scripts/          generate_all.sh, analyse_all.sh - the whole run, resumable
src/parallel.py   CPU worker resolution (no GPU — see below)
tests/            62 gate tests + 210 specification tests
docs/             WORKSTREAMS.md — who owns which file
```

Import with the top-level name: `from corpus.schemas import Query`.

## Performance and hardware

Runs on any machine — Apple silicon, x86, cluster node — with no device
configuration. `DRIFTWATCH_N_WORKERS` sets the worker count; `0` (the default)
auto-detects and leaves one core free.

**There is no CUDA/MPS/CPU switch in driftwatch, and that is deliberate.**
Models run inside the Ollama server, which uses the GPU on its own - Metal on
Apple silicon, CUDA on NVIDIA. driftwatch talks to it over HTTP and never loads
a model itself (C7); what executes in its own process is numpy/scipy over text
statistics. Reaching for `torch` would trip the C7 gate.

The one heavy local computation is the calibration bootstrap — 20,000 windows
x 7 signals x 800 completions. It takes seconds, not hours, because no window
is ever materialised: each signal reduces a query's k samples to sufficient
statistics ONCE (`signals/groups.py`), and a bootstrap window is scored as an
index array into that table. `score(window)` calls the same aggregation, and
calibration cross-checks the two paths on its first windows before trusting
them. Each window's draw has its own seed (`parallel.derive_worker_seeds`), so
the numbers cannot depend on how the work is split.

## Configuration

`base_url` and `api_key` have no defaults - set `DRIFTWATCH_BASE_URL` and
`DRIFTWATCH_API_KEY`, or use a `.env` (locally: `http://localhost:11434/v1` and
any non-empty key). Model ids live only in `src/config/models.yaml`, each with
the library digest it is pinned to; any three-rung ladder from one model family
substitutes directly - change the ids and pins there and nothing else.

## Decisions recorded in code

Choices the spec left open, fixed once and documented where they live:

| Decision | Where |
|---|---|
| Dataset sources, pinned to exact Hub commits; `eli5` and `banking77` substitutes | `corpus/build.py` |
| squad2 unanswerables in scope at their natural rate (~50%) | `corpus/build.py` |
| XSum documents capped at 800 words; cp1252 mojibake repaired (~11% of rows) | `corpus/build.py` |
| S1 length unit: whitespace tokens | `signals/length.py` |
| S3: a parse-failed sample agrees with nothing | `signals/consistency.py` |
| S4: MATTR (window 25) + trigram repeat rate, one-sided, equal weights | `signals/lexical.py` |
| S5 strips one code fence: measures schema conformance, not fence avoidance | `signals/fmt.py` |
| S6: Miller-Madow entropy; baseline measured at window size | `signals/entropy.py` |
| S7: cosine on unit vectors; parse failures excluded symmetrically | `signals/embedding.py` |
| Thresholds: GPD peaks-over-threshold when n x alpha' < 10 (always, at 20,000) | `detect/calibrate.py` |
| Replay latency counted from the ONSET window, not the first fully faulty one | `evaluate/replay.py` |

**S7 offline stand-in.** With no embedding vectors injected, S7 falls back to a
feature-hashed bag of words so it can run in tests. That is string hashing, not
a model, but it measures lexical rather than semantic drift. The CLI never uses
it on real data: `calibrate` and `evaluate` inject vectors from the embedding
round and refuse to run S7 without them.

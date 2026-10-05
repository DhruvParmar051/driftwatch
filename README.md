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
uv run pytest -m spec    # 139 behavioural specs, incl. the Stage 0-7 gates
```

The four freeze tests require a **clean git tree**: a freeze stamps the current
commit and refuses to point at code that was never committed. Commit, then run.

What has NOT been run is a real round on the backend. Everything below Stage 2
is exercised against synthetic data and an in-memory batch backend
(`tests/fake_backend.py`); the numbers mean the pipeline works, nothing yet
about real models.

## Running a round

```bash
uv run driftwatch corpus build                       # 5,000 queries + splits (HuggingFace)
uv run driftwatch probe --rows 20                    # STOP if this fails - see below
uv run driftwatch expand --condition healthy --round-id r001 --out data/r001.jsonl
uv run driftwatch submit data/r001.jsonl --condition healthy --round-id r001 --chunks 4
uv run driftwatch resume r001                        # rerun freely after any crash
uv run driftwatch embed --condition healthy --round-id e001 && uv run driftwatch resume e001
# ... same for F1, F2, F3 (and F5). F4 is derived from healthy rows - nothing to generate.
uv run driftwatch calibrate --completions data/completions.parquet
git commit ...                                       # freeze refuses a dirty tree
uv run driftwatch freeze --thresholds data/thresholds.proposed.json --completions data/completions.parquet
uv run driftwatch evaluate --completions data/completions.parquet
uv run driftwatch report --results data/results/evaluation.json --out-dir data/report
```

`submit` splits a round into token-balanced jobs and writes a manifest before
uploading anything; `resume` recovers orphaned jobs, collects what has landed,
and reconciles every chunk and then the whole round. `driftwatch evaluate
--synthetic` runs Stages 5-7 end to end on synthetic data, without a freeze.

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

Submits one prompt x 10 distinct seeds and x 10 identical seeds. The platform
documents `seed` as honoured, but that documentation is explicitly doc-derived
and unverified against a live server.

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
| C7 | No local model inference - embeddings come from the endpoint | *enforced* |
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
src/corpus/       query construction and splits  (code; data lives in data/)
src/provider/     batch client, expand, submit, collect, manifest (resume)
src/signals/      windows, groups (shared scoring base) + S1..S7
src/detect/       calibrate, detectors, freeze
src/oracle/       ground-truth scoring (never imported by detect/)
src/evaluate/     replay, metrics, bootstrap, pipeline (the protocol, step by step)
src/synthetic/    offline completion generator
src/parallel.py   CPU worker resolution (no GPU — see below)
tests/            61 gate tests + 139 specification tests
docs/             WORKSTREAMS.md — who owns which file
```

Import with the top-level name: `from corpus.schemas import Query`.

## Performance and hardware

Runs on any machine — Apple silicon, x86, cluster node — with no device
configuration. `DRIFTWATCH_N_WORKERS` sets the worker count; `0` (the default)
auto-detects and leaves one core free.

**There is no CUDA/MPS/CPU switch, and that is deliberate.** No model runs
locally: chat completions and embeddings both come from the batch endpoint
(C7). What executes here is numpy/scipy over text statistics, which does not
use a GPU. Reaching for one would mean adding `torch`, which trips the C7 gate.

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
`DRIFTWATCH_API_KEY`, or use a `.env`. Model ids live only in
`src/config/models.yaml`; any three-rung ladder from one family on any
OpenAI-compatible batch backend substitutes directly.

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

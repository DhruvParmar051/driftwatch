# driftwatch

Label-free detection of silent quality degradation in deployed language models.

A monitor that detects quality degradation from the statistical shape of
generated text alone - no labels at detection time - plus the evaluation harness
that proves it works. Four faults are injected at known onsets; the monitor is
calibrated on healthy data only, frozen, then scored on detection latency and
false-alarm rate.

## Status

**Scaffold.** Architecture is complete; the research work is stubbed. Every
unimplemented function raises `NotImplementedError` and has a docstring
specifying its behaviour, constraints and edge cases.

```bash
uv sync
uv run pytest -m gate    # 36 architectural gates - must always pass
uv run pytest -m spec    # remaining work; the failure count is the burndown
```

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
src/config/       models.yaml, refusal_patterns.yaml, jsonx_schema.json
src/corpus/       query construction and splits  (code; data lives in data/)
src/provider/     batch client, expand, submit, collect
src/signals/      windows + S1..S7
src/detect/       calibrate, detectors, freeze
src/oracle/       ground-truth scoring (never imported by detect/)
src/evaluate/     replay, metrics, bootstrap
src/synthetic/    offline completion generator
tests/            37 gate tests + 112 specification tests
docs/             WORKSTREAMS.md — who owns which file
```

Import with the top-level name: `from corpus.schemas import Query`.

## Configuration

`base_url` and `api_key` have no defaults - set `DRIFTWATCH_BASE_URL` and
`DRIFTWATCH_API_KEY`, or use a `.env`. Model ids live only in
`src/config/models.yaml`; any three-rung ladder from one family on any
OpenAI-compatible batch backend substitutes directly.

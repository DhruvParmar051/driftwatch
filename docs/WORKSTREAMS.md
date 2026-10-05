# Workstreams

> **Status on branch `dhruv`:** every file in the table below is implemented and the full test suite passes. The ownership table records the original split of the work; it no longer describes outstanding stubs.

Four owners, 25 files, 116 stubs. Dhruv carries the silent-failure work; the other three are exactly balanced. All paths below are relative to `src/`. Ownership is **file-level and exclusive**: every
file has exactly one owner, so two people never edit the same file. Read your
column, then read the dependency rules - they decide the order, not preference.

## Ownership

Deliberately uneven. Dhruv takes the work where a mistake is **silent** — where
the pipeline still runs, the numbers still look plausible, and the result is
quietly invalid. The other three are balanced to exactly 21 stubs each.

| Owner | Files | Stubs |
|---|---|---|
| **Dhruv** — provider, calibration & orchestration | `provider/client.py`, `provider/catalogue.py`, `provider/expand.py`, `provider/submit.py`, `provider/collect.py`, `provider/manifest.py`, `detect/calibrate.py`, `detect/freeze.py`, `cli.py` | **53** |
| **Pronnati** — corpus & evaluation | `corpus/build.py`, `corpus/splits.py`, `evaluate/metrics.py`, `evaluate/bootstrap.py`, `evaluate/replay.py` | 21 |
| **Mahak** — windows, semantic signals & detection | `signals/windows.py`, `signals/consistency.py`, `signals/entropy.py`, `signals/embedding.py`, `detect/detectors.py` | 21 |
| **Aditya** — synthetic, surface signals & oracle | `synthetic/generate.py`, `signals/length.py`, `signals/refusal.py`, `signals/lexical.py`, `signals/fmt.py`, `oracle/score.py` | 21 |

Nobody owns `settings.py`, `corpus/schemas.py`, `signals/base.py`,
`signals/registry.py`, `log.py`, `parallel.py`, `version.py`, the config
YAML/JSON, or anything under `tests/`. Those are **frozen shared contracts** —
see Rules below.

### Why Dhruv's files are the dangerous ones

Every file below fails *quietly*. Nothing crashes; the study is just wrong.

| File | How it fails silently |
|---|---|
| `detect/calibrate.py` | Calibrating on anything but `calib` inflates **every** detection number, and nothing in the output looks wrong. The single most dangerous function in the project. |
| `detect/freeze.py` | If the row hash or the commit check is weak, C4 and C5 stop being enforceable and "calibrated on healthy data only" becomes an unverifiable claim. |
| `provider/expand.py` | Three separate silent killers: `n>1` passes submission and fails at the worker 18 hours later; a duplicate seed collapses S3 to a constant 1.0; strict `json_schema` mode makes the platform do S5's job and S5 reads zero forever. |
| `provider/collect.py` | The daemon *skips malformed input lines*. Without real set-based reconciliation a round comes back short and every per-cell sample size is wrong — invisibly. |
| `provider/manifest.py` | Resubmitting an orphaned chunk wastes a day of GPU and returns duplicate rows; skipping one leaves the round permanently short. Both unrecoverable after the fact. |
| `provider/catalogue.py` | A silently substituted model makes the entire healthy-vs-fault comparison meaningless. |
| `provider/client.py`, `submit.py` | All I/O for the project. `submit` stamps the idempotency key `manifest` matches on — one owner keeps that contract intact. |
| `cli.py` | Thin wiring, but the one file all four would otherwise edit: the project's only real merge-conflict risk. |

### The cost of this split — know it going in

- **Dhruv holds 46% of the project and 100% of `provider/`.** He is the only
  person who can submit or collect a round. If he stalls, no real data arrives
  and the other three run out of work at Stage 4.
- **His queue is back-loaded.** `cli.py` needs every other module to exist
  first, so the largest single file is also the last.
- **Mitigation:** the moment `expand.py`, `submit.py` and `collect.py` pass
  their spec tests, walk a second person through submitting a round. A bus
  factor of one on the only path to data is the real risk here, not the stub
  count.

### Why the other three fall as they do

- **Pronnati owns corpus and evaluation end to end.** `build` and `splits`
  belong together (splits is the leakage boundary the corpus defines), and
  `replay`/`metrics`/`bootstrap` are one loop.
- **Mahak pairs `detectors` with `windows` and the semantic signals.** The
  alarm rules consume window scores directly.
- **Aditya owns `oracle/`, and no `detect/` owner does.** C3 forbids `detect/`
  from reaching `oracle/`; `detect/` is split between Dhruv and Mahak, so
  putting the oracle with neither keeps that a people boundary as well as a CI
  gate. He keeps the fewest files on the critical path because two people wait
  on `synthetic/generate.py`.

## Dependency order

Only two files gate other people's work:

```
  Aditya: synthetic/generate.py ─┐
                                 ├─► all 7 signals ──► detect/calibrate.py ──► evaluate/replay.py
  Mahak:  signals/windows.py ────┘   (Aditya, Mahak)      (Mahak)                 (Dhruv)
```

**Both must land by end of day 2.** They are each day-1 startable - neither
depends on anything unfinished - so there is no reason for them to slip. Until
they do, five other files are blocked.

### Day 1 — everyone starts, nobody waits

| Owner | Start with | Why it is unblocked |
|---|---|---|
| Dhruv | `provider/expand.py`, `provider/client.py` | Mocked `httpx` transport; no network, no credentials |
| Pronnati | `evaluate/metrics.py`, `evaluate/bootstrap.py` | Pure functions over floats, dicts and arrays |
| Aditya | **`synthetic/generate.py`** | Needs only `Query` objects, which you can hand-build |
| Mahak | **`signals/windows.py`**, `detect/detectors.py` | Windows need only `Query`/`Completion`; detectors are pure numerics |

### After the critical path lands

| Owner | Then |
|---|---|
| Dhruv | `submit.py`, `collect.py`, `manifest.py`, `catalogue.py`, then `calibrate.py`, `freeze.py`, and `cli.py` last |
| Pronnati | `corpus/build.py`, `corpus/splits.py`, then `evaluate/replay.py` last |
| Aditya | `length.py`, `refusal.py`, `lexical.py`, `fmt.py`, then `oracle/score.py` |
| Mahak | `consistency.py`, `entropy.py`, `embedding.py`, `detectors.py` |

`evaluate/replay.py` (Pronnati) and `cli.py` (Dhruv) are genuinely last: replay
needs windows, synthetic data, signals and frozen thresholds, and each CLI
command needs its module. Pronnati fills that wait with `corpus/build.py` — the
largest single file in the project, six datasets to normalise — which blocks
nobody. Dhruv fills it with the rest of `provider/`.

## Rules

**0. Prompts live in `config/prompts.yaml`, never in source.** Slice templates
and both system prompts (healthy, and the corrupted one that *is* fault F3) are
there, versioned. The version is recorded at freeze time and checked by
`freeze.verify()`, because thresholds describe model behaviour on those exact
prompts. `tests/test_prompts.py` fails the build on an inlined prompt.

The F3 prompt must be **pre-registered** - fixed before the fault round is
submitted, never adjusted after seeing detection results. Tuning it to make F3
more detectable is the same error as tuning a threshold after the freeze.

**1. Shared contracts are frozen.** `settings.py`, `corpus/schemas.py`,
`signals/base.py` and `signals/registry.py` are what stop four people drifting
apart. Changing a field breaks all four tracks at once. Propose the change to
the group; do not edit unilaterally.

**2. Spec tests are read-only.** The 112 tests under `tests/` marked `spec` are
the shared definition of done. If one looks wrong, it is either a real design
question or a genuine bug in the test - raise it with the group. Editing a
failing test so it passes is how a constraint quietly disappears.

**3. Gates must stay green.** `uv run pytest -m gate` must pass on every commit.
If a gate goes red, a constraint has been violated - stop and fix it. Never
weaken the gate.

**4. Your work is done when your spec tests pass.**

```bash
uv run pytest -m spec -q                      # burndown across the project
uv run pytest tests/test_signals_spec.py -q   # your area
```

**5. `tests/test_signals_spec.py` covers two owners** (Aditya's four signals and
Mahak's three). It is read-only for both, so there is no conflict - but a green
run needs both halves, so check with each other before claiming the signals gate.

**6. The freeze is a one-time, witnessed act.** Mahak writes `freeze.py`; Dhruv
runs `driftwatch freeze` once, with Mahak present. After it, no threshold
changes - not to make a gate pass, not to fix an obviously wrong number. If the
method is wrong, fix the method and refreeze from a new commit (C4).

## Things that will cost a day if you get them wrong

Each is documented in the relevant docstring; they are collected here because
they are the failures that do not announce themselves.

- **`n > 1` is not caught at submission** (Pronnati). The platform's validator
  never inspects it. The row passes, reaches a worker, and fails *there* -
  scattered failed rows, 18 hours later, one wasted round.
- **Identical seeds collapse S3 to a constant 1.0** (Aditya, Pronnati). If
  self-consistency reads 1.0 across a whole window, suspect duplicate seeds
  before believing the model.
- **`jsonx` must not request strict `json_schema` mode** (Pronnati, Aditya).
  The platform would validate server-side and fail the row, so malformed JSON
  never returns as text and S5 silently reads zero forever.
- **Unstratified windows starve S5 and S6** (Mahak). Both are computed on a
  single slice. Without fixed composition their scores are not comparable
  between windows, and the thresholds get set by sampling noise.
- **Calibrating on anything but `calib` inflates every result invisibly**
  (Mahak). `assert_calib_only` exists for this; call it first, every time.
- **Never resubmit a chunk left `submitting`** (Pronnati). A crash between
  creating a batch and recording its id leaves a job that may be running right
  now. Resubmitting wastes a day of GPU and returns duplicate rows. Match on
  the stamped idempotency key first — `manifest.recover_orphans`.
- **A non-detection is not a large latency** (Dhruv). Recording the stream
  length instead of `None` turns a failure into a finite number and hides it in
  the mean.

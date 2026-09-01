# Workstreams

Four owners, 25 files, 116 stubs. All paths below are relative to `src/`. Ownership is **file-level and exclusive**: every
file has exactly one owner, so two people never edit the same file. Read your
column, then read the dependency rules - they decide the order, not preference.

## Ownership

| Owner | Files | Stubs |
|---|---|---|
| **Dhruv** — corpus, evaluation & orchestration | `corpus/build.py`, `corpus/splits.py`, `evaluate/replay.py`, `evaluate/metrics.py`, `evaluate/bootstrap.py`, `oracle/score.py`, `cli.py` | **35** |
| **Pronnati** — provider | `provider/client.py`, `provider/catalogue.py`, `provider/expand.py`, `provider/submit.py`, `provider/collect.py`, `provider/manifest.py` | 34 |
| **Mahak** — windows, semantic signals & detection | `signals/windows.py`, `signals/consistency.py`, `signals/entropy.py`, `signals/embedding.py`, `detect/calibrate.py`, `detect/freeze.py`, `detect/detectors.py` | 30 |
| **Aditya** — synthetic & surface signals | `synthetic/generate.py`, `signals/length.py`, `signals/refusal.py`, `signals/lexical.py`, `signals/fmt.py` | 17 |

Nobody owns `settings.py`, `corpus/schemas.py`, `signals/base.py`,
`signals/registry.py`, `log.py`, `parallel.py`, `version.py`, the config
YAML/JSON, or anything under `tests/`. Those are **frozen shared contracts** —
see Rules below.

### Why the split falls this way

- **Dhruv owns `cli.py` alone.** Every command is thin wiring over someone
  else's module, so it is the one file all four people would otherwise touch —
  and the project's only real merge-conflict risk. One owner removes it. Write
  each command once its underlying module lands.
- **Dhruv owns `oracle/` and Mahak owns all of `detect/`.** C3 forbids
  `detect/` from importing `oracle/`, and keeping them under different owners
  makes that a people boundary as well as a CI gate. `evaluate/` *is* allowed to
  use the oracle — that is exactly what validates label-free detection against
  ground truth — so oracle sits naturally with whoever owns evaluation.
- **Mahak pairs `detectors` with `calibrate` and `freeze`.** Thresholds feed
  the alarm rules directly; splitting them invites a threshold tuned between
  estimating it and using it.
- **Pronnati keeps `submit` and `manifest` together.** `submit_batch` stamps
  the idempotency key that `recover_orphans` matches on. Separate owners for
  those two is how that contract quietly breaks.
- **Aditya carries the fewest stubs deliberately.** He owns the critical path
  and two other people wait on him. Loading him to parity slows everyone down.

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
| Dhruv | `evaluate/metrics.py`, `evaluate/bootstrap.py` | Pure functions over floats, dicts and arrays |
| Pronnati | `provider/expand.py`, `provider/client.py` | Mocked `httpx` transport; no network, no credentials |
| Aditya | **`synthetic/generate.py`** | Needs only `Query` objects, which you can hand-build |
| Mahak | **`signals/windows.py`**, `detect/freeze.py`, `detect/detectors.py` | Windows need only `Query`/`Completion`; freeze needs only `Completion`; detectors are pure numerics |

### After the critical path lands

| Owner | Then |
|---|---|
| Dhruv | `corpus/build.py`, `corpus/splits.py`, `oracle/score.py`, then `evaluate/replay.py` and `cli.py` last |
| Pronnati | `submit.py`, `collect.py`, `manifest.py`, `catalogue.py`, then `oracle/score.py` |
| Aditya | `length.py`, `refusal.py`, `lexical.py`, `fmt.py` |
| Mahak | `consistency.py`, `entropy.py`, `embedding.py`, then `calibrate.py` last |

`evaluate/replay.py` and `cli.py` are genuinely last: replay needs windows,
synthetic data, signals and frozen thresholds, and each CLI command needs its
module. Dhruv fills that wait with `corpus/build.py` — the largest single file
in the project, six datasets to normalise — which blocks nobody.

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

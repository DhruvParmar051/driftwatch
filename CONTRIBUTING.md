# Contributing

Four people are implementing this, probably with help from four different AI
assistants. Without a shared contract that produces four house styles in one
repository, and reconciling them afterwards costs more than writing the code did.

This document is the contract. **Give it to your assistant along with the stub
file you are implementing.**

---

## The one-paragraph version

Every function you implement already has its **name, typed signature, and a
docstring specifying its behaviour, constraints and edge cases**. Your job is to
write the body and nothing else. Do not change the signature. Do not rewrite the
docstring. Do not edit the tests. If the docstring seems wrong, raise it with
the group — it is the shared specification, and changing it unilaterally is how
four people end up building four different things.

---

## Paste this to your assistant

Copy everything between the lines, then paste the stub file below it.

```
You are implementing one function in an existing Python 3.11 research codebase.
Follow these rules exactly. They are not stylistic preferences.

STRUCTURE
- Modules live directly under src/ and are imported by TOP-LEVEL name:
      from corpus.schemas import Query, Completion
      from settings import get_settings
      from log import get_logger
  There is NO package prefix. `from driftwatch.corpus...` is WRONG and will not
  import. `from src.corpus...` is also WRONG.
- Never use relative imports (`from .foo import`). Absolute only.
- Do not create new files. Do not add dependencies. Do not add a __main__ block.

WHAT YOU MAY CHANGE
- ONLY the function body (replace `raise NotImplementedError`).
- Never change the function name, parameters, type hints, or return type.
- Never rewrite or shorten the docstring. Keep it exactly as given.
- Never modify anything in tests/.

STYLE
- `from __future__ import annotations` at the top of every module (already there).
- Full type annotations on every parameter and return value.
- No bare `except:` and no `except Exception:` without re-raising. Catch the
  specific exception you expect.
- Never use print(). Use: log = get_logger(__name__); log.info("event", key=value)
- Use pathlib.Path, not os.path.
- Line length 100.
- Raise on invalid input. Do not return None, -1, or an empty result as an error
  signal — a silent wrong answer is the failure mode this project exists to study.

CORRECTNESS RULES THAT OVERRIDE CONVENIENCE
- The docstring's "Constraints" and "Edge cases" sections are requirements.
  Implement them literally, including the ones that look like over-caution.
- Determinism: identical inputs must produce byte-identical output. Never use
  the global `random` module or an unseeded numpy RNG. Thread a seeded
  np.random.default_rng(seed) through.
- Never read configuration from a literal. Model ids, prompts, refusal patterns
  and thresholds come from settings.py or src/config/*.yaml.

OUTPUT
- Return the complete function, docstring unchanged, body implemented.
- No explanation unless I ask. No markdown fences around the whole file.
```

---

## Verify before you push

Three commands. All three must pass. They are the same checks CI runs, so if
they pass locally they pass in CI, whoever wrote the code.

```bash
uv run ruff check . && uv run ruff format .
```

```bash
uv run pytest -m gate -q
```

61 gate tests. **These must always pass.** A red gate means a project constraint
has been violated — stop and fix it rather than working around it.

```bash
uv run pytest tests/test_<your_area>_spec.py -q
```

Your work is done when your spec tests pass. That is the definition of done, not
"it looks right".

---

## Why the style is enforced by a tool

`ruff check` and `ruff format --check` run in CI and fail the build. This is
deliberate: a style guide that relies on everyone remembering it does not
survive four people, four assistants and a deadline. The tool does not care
which model wrote the code, so the result is identical either way.

Run `uv run ruff format .` before every commit and formatting stops being
something anyone has to think about.

---

## Things assistants get wrong on this codebase

Every one of these has a mechanical check, so you will find out — but knowing
in advance saves a round trip.

| What it will do | What is correct |
|---|---|
| `from driftwatch.corpus.schemas import Query` | `from corpus.schemas import Query` — there is no package prefix |
| `import logging` for the project logger | `from log import get_logger` — `logging` is the stdlib |
| Wrap everything in `try/except Exception` | Let it raise. Catch only what you expect. |
| Drop rows that fail to parse | Keep them with `parse_ok=False` (C6) — parse failure is a signal |
| `print()` progress | `log.info("event", key=value)` |
| Add `torch` or `sentence-transformers` for embeddings | Forbidden (C7). Embeddings come from the batch endpoint. |
| Inline a prompt or a model id | They live in `src/config/*.yaml` |
| Use `random.shuffle` or `np.random.shuffle` | Seeded generator, threaded through — output must be reproducible |
| Return `None` when something is wrong | Raise. A silent wrong answer is the bug class this project studies. |
| "Simplify" a docstring constraint that looks paranoid | It is load-bearing. Ask before removing. |

---

## Review checklist

Before asking for a review, confirm:

- [ ] `ruff check` and `ruff format --check` pass
- [ ] `pytest -m gate` — all 61 pass
- [ ] My spec tests pass
- [ ] Signature and docstring are byte-identical to the stub
- [ ] Nothing under `tests/` changed
- [ ] No new files, no new dependencies
- [ ] No model id, prompt, URL or threshold written as a literal
- [ ] Same input twice gives the same output

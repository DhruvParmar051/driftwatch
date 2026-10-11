"""Threshold freezing and integrity verification. Track T4.

`thresholds.json` is the experimental record. Once written it is immutable: it
carries the git commit and a hash of the exact calibration rows it was derived
from, so that any later claim of "these thresholds were calibrated on healthy
data only" is checkable rather than asserted.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from corpus.schemas import Completion, Thresholds
from log import get_logger
from settings import PROJECT_ROOT, load_prompts, load_refusal_patterns

log = get_logger(__name__)

#: The row fields that define a calibration row's CONTENT. Timestamps are
#: transport metadata: they do not change what was calibrated on, and their
#: precision is the first thing a parquet round trip alters.
HASHED_FIELDS: tuple[str, ...] = (
    "query_id",
    "sample_idx",
    "condition",
    "seed",
    "model_id",
    "digest",
    "digest_source",
    "text",
    "finish_reason",
    "parse_ok",
)


class FrozenIntegrityError(RuntimeError):
    """The frozen file does not match the data or code it claims to describe.

    Never catch this to continue. It means the thresholds are not what they say
    they are, and every number computed downstream is unattributable.
    """


def hash_calibration_rows(completions: list[Completion]) -> str:
    """Stable 'sha256:...' digest over the exact rows used for calibration.

    Constraints:
        Must be order-independent and process-independent: sort by a stable key
        before hashing, and never use Python's salted `hash()`. Hash the row
        CONTENT, not the file - the same rows written by a different pandas
        version must produce the same digest, or verify() fails spuriously and
        the team learns to ignore it.
    """
    records = sorted(
        json.dumps(
            [getattr(c, f) for f in HASHED_FIELDS], ensure_ascii=False, separators=(",", ":")
        )
        for c in completions
    )
    h = hashlib.sha256()
    for record in records:
        h.update(record.encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def current_git_commit() -> str:
    """Short git commit hash of the working tree.

    Raises:
        FrozenIntegrityError: if the tree is DIRTY. Freezing from uncommitted
            code produces a record pointing at a commit that never contained the
            calibration logic actually used.

    """

    def git(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise FrozenIntegrityError(f"cannot read git state: {exc}") from exc
        return out.stdout.strip()

    dirty = git("status", "--porcelain", "--untracked-files=normal")
    if dirty:
        raise FrozenIntegrityError(
            "working tree is dirty - commit first, so the freeze points at the code that "
            f"actually produced it:\n{dirty}"
        )
    return git("rev-parse", "--short=12", "HEAD")


def write(
    path: Path,
    thresholds: dict[str, float],
    calib_completions: list[Completion],
    family_wise_budget: float,
    per_signal_alpha: float,
    k: int,
    window_size: int,
    window_stride: int,
    bootstrap_windows: int,
    calibration_seed: int,
    scoring: str = "pooled",
) -> Thresholds:
    """Write thresholds.json. THE POINT OF NO RETURN.

    Behaviour:
        Assemble a Thresholds record - commit hash, calibration-row hash, window
        geometry, the Sidak level actually used, the refusal-patterns version -
        and write it.

    Raises:
        FrozenIntegrityError: if `path` already exists. Overwriting a freeze is
            how a recalibration hides itself. To refreeze deliberately, move the
            old file aside by hand, so the act is visible in the shell history
            and in git.

    Constraints:
        C4 - after this returns, nothing may change a threshold. Not to make a
        gate pass, not to fix an obviously-wrong number. Fix the method and
        refreeze from scratch under a new commit.

    """
    if path.exists():
        raise FrozenIntegrityError(
            f"{path} already exists. Refreezing is deliberate: move the old file aside by hand."
        )
    record = Thresholds(
        frozen_at=datetime.now(UTC),
        git_commit=current_git_commit(),
        calib_rows_hash=hash_calibration_rows(calib_completions),
        k=k,
        window_size=window_size,
        window_stride=window_stride,
        family_wise_budget=family_wise_budget,
        per_signal_alpha=per_signal_alpha,
        n_signals=len(thresholds),
        bootstrap_windows=bootstrap_windows,
        calibration_seed=calibration_seed,
        refusal_patterns_version=str(load_refusal_patterns()["version"]),
        prompts_version=str(load_prompts()["version"]),
        thresholds=dict(thresholds),
        scoring=scoring,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    # os.link refuses to overwrite: a concurrent freeze cannot clobber this one.
    try:
        os.link(tmp, path)
    except FileExistsError as exc:
        raise FrozenIntegrityError(f"{path} appeared while freezing - not overwriting") from exc
    finally:
        tmp.unlink(missing_ok=True)
    log.info(
        "freeze.written", path=str(path), commit=record.git_commit, rows_hash=record.calib_rows_hash
    )
    return record


def read(path: Path) -> Thresholds:
    """Read and parse thresholds.json. Raises if absent or malformed."""
    raw = path.read_text(encoding="utf-8")
    try:
        return Thresholds.model_validate_json(raw)
    except ValueError as exc:
        raise FrozenIntegrityError(f"{path} is not a valid thresholds record: {exc}") from exc


def verify(path: Path, calib_completions: list[Completion]) -> Thresholds:
    """Verify the frozen file against the calibration data. Raises on mismatch.

    Behaviour:
        Read the record, recompute `calib_rows_hash` from `calib_completions`,
        and compare. Also check that the current refusal-patterns version
        matches the frozen one - if the patterns changed, the frozen refusal
        threshold is expressed in units that no longer exist. Same for the
        prompts version: thresholds describe behaviour on specific prompts.

    Raises:
        FrozenIntegrityError: on any mismatch. `evaluate/` calls this and
            refuses to run if it raises.

    Constraints:
        This is what makes C4 and C5 checkable rather than merely stated. A
        mutated calibration set MUST fail here - that is the Stage 5 gate.

    """
    record = read(path)
    problems = []
    actual_hash = hash_calibration_rows(calib_completions)
    if actual_hash != record.calib_rows_hash:
        problems.append(
            f"calibration rows changed: frozen {record.calib_rows_hash}, now {actual_hash}"
        )
    patterns = str(load_refusal_patterns()["version"])
    if patterns != record.refusal_patterns_version:
        problems.append(
            f"refusal patterns are v{patterns}, thresholds were frozen on "
            f"v{record.refusal_patterns_version}"
        )
    prompts = str(load_prompts()["version"])
    if prompts != record.prompts_version:
        problems.append(f"prompts are v{prompts}, frozen on v{record.prompts_version}")
    if problems:
        raise FrozenIntegrityError(
            f"{path} does not describe this data:\n  " + "\n  ".join(problems)
        )
    log.info("freeze.verified", path=str(path), commit=record.git_commit)
    return record

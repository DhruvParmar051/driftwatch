"""Threshold freezing and integrity verification. STUB - track T4.

`thresholds.json` is the experimental record. Once written it is immutable: it
carries the git commit and a hash of the exact calibration rows it was derived
from, so that any later claim of "these thresholds were calibrated on healthy
data only" is checkable rather than asserted.
"""

from __future__ import annotations

from pathlib import Path

from corpus.schemas import Completion, Thresholds


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
    raise NotImplementedError


def current_git_commit() -> str:
    """Short git commit hash of the working tree.

    Raises:
        FrozenIntegrityError: if the tree is DIRTY. Freezing from uncommitted
            code produces a record pointing at a commit that never contained the
            calibration logic actually used.
    """
    raise NotImplementedError


def write(path: Path, thresholds: dict[str, float], calib_completions: list[Completion],
          family_wise_budget: float, per_signal_alpha: float, k: int, window_size: int,
          window_stride: int, bootstrap_windows: int, calibration_seed: int) -> Thresholds:
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
    raise NotImplementedError


def read(path: Path) -> Thresholds:
    """Read and parse thresholds.json. Raises if absent or malformed."""
    raise NotImplementedError


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
    raise NotImplementedError

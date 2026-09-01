"""Result retrieval, parsing and reconciliation. STUB - track T2.

C10 lives here, and it is not a theoretical precaution: the worker daemon
SKIPS malformed input lines with a warning and simply does not execute them, so
a bad row vanishes from the output rather than returning an error. Submitted
and returned counts genuinely can differ.

A partially returned batch breaks every per-cell sample size in
the experiment, and does so invisibly: the analysis still runs, the numbers
still look plausible, and every window is quietly built on fewer rows than it
claims. Reconciliation must therefore FAIL LOUDLY, never warn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from corpus.schemas import Completion


class ReconciliationError(RuntimeError):
    """Returned rows do not match submitted rows. Fail the round; do not proceed."""


@dataclass
class ReconciliationReport:
    """Set comparison of submitted vs returned custom_ids."""

    n_submitted: int = 0
    n_returned: int = 0
    missing: list[str] = field(default_factory=list)     # submitted, never came back
    unexpected: list[str] = field(default_factory=list)  # returned, never submitted
    duplicated: list[str] = field(default_factory=list)  # returned more than once
    n_parse_failures: int = 0

    @property
    def is_clean(self) -> bool:
        """True only if all three discrepancy lists are empty AND counts match."""
        raise NotImplementedError

    def raise_if_dirty(self) -> None:
        """Raise ReconciliationError with a full breakdown unless `is_clean`.

        Constraints:
            C10 - this raises. It does not log a warning and continue. A
            downstream analysis on a partial batch is worse than no analysis,
            because it looks like a result.
        """
        raise NotImplementedError


def parse_result_line(line: str) -> tuple[str, str, str, bool]:
    """Parse one line of the batch output JSONL.

    Returns:
        (custom_id, text, finish_reason, parse_ok).

    Behaviour:
        Extract the completion text from the response envelope. Set
        parse_ok=False for: an error response, a missing/empty choices array,
        empty or whitespace-only content, or a malformed envelope.

    Constraints:
        C6 - a parse failure returns a row with parse_ok=False and text="". It
        NEVER returns None and the caller NEVER drops it. Parse failure is a
        degradation signal: a model that has got worse produces more of them,
        and discarding them removes the very evidence being looked for.

    Raises:
        ValueError: only if `custom_id` itself is unrecoverable - that is a
            reconciliation problem (the row cannot be matched to a query), not
            a parse failure.
    """
    raise NotImplementedError


def collect_results(output_path: Path, submitted_custom_ids: set[str], model_id: str,
                    digest: str, digest_source: str, submitted_at) -> tuple[list[Completion], ReconciliationReport]:
    """Parse a downloaded output file into Completions and reconcile it.

    Behaviour:
        Parse every line, decode each custom_id back into
        (query_id, sample_idx, condition, round_id) via
        `expand.decode_custom_id`, rebuild Completion rows stamped with the
        round's model_id and digest (C9), and compare the returned id set
        against `submitted_custom_ids`.

    Returns:
        (completions, report). The caller MUST call `report.raise_if_dirty()`.
        Returning the report rather than raising here is deliberate: a caller
        may want to persist the partial rows for diagnosis before failing.

    Constraints:
        C6 - every submitted row appears in the output, including failures.
        C10 - the report is a set comparison, not a count comparison. Equal
              counts with one duplicate and one missing row is a real failure
              mode and a length check would pass it.
    """
    raise NotImplementedError


def write_completions(completions: list[Completion], path: Path) -> int:
    """Append/write completions to parquet. Returns rows written.

    Constraints:
        Column order and dtypes must match schemas.Completion exactly, so that
        re-reading and re-validating every row succeeds. That round-trip is the
        Stage 0 gate and is what lets synthetic and real rows be
        indistinguishable to every downstream stage.
    """
    raise NotImplementedError


def read_completions(path: Path) -> list[Completion]:
    """Read and re-validate a completions parquet. Raises on any invalid row."""
    raise NotImplementedError

"""Data contracts. FULLY IMPLEMENTED - the shared vocabulary of the project.

Every track imports from here. Changing a field in this module is a breaking
change for all four workstreams, so agree edits before making them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from settings import VERIFIABLE_SLICES


class Query(BaseModel):
    """One row of `queries.jsonl`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str = Field(description="Stable id, '<slice>_<5-digit index>', e.g. 'gsm8k_00042'.")
    slice: str
    prompt: str
    ground_truth: str | None = Field(
        default=None, description="Non-null exactly on the verifiable slices."
    )
    label_space: list[str] | None = Field(
        default=None, description="Non-null only for banking77 (its 77 intent labels)."
    )
    split: str

    @model_validator(mode="after")
    def _check_slice_invariants(self) -> Query:
        if self.slice in VERIFIABLE_SLICES and self.ground_truth is None:
            raise ValueError(
                f"{self.query_id}: slice {self.slice!r} is verifiable but ground_truth is None"
            )
        if self.slice not in VERIFIABLE_SLICES and self.ground_truth is not None:
            raise ValueError(
                f"{self.query_id}: slice {self.slice!r} is not verifiable but has ground_truth"
            )
        if (self.slice == "banking77") != (self.label_space is not None):
            raise ValueError(f"{self.query_id}: label_space must be set iff slice is banking77")
        return self


class Completion(BaseModel):
    """One row of `completions.parquet` - one sample of one query.

    There are k rows per (query_id, condition), one per sample_idx, each with a
    DISTINCT seed (C2). Identical seeds return identical text and would
    collapse self-consistency (S3) to a constant 1.0.
    """

    model_config = ConfigDict(extra="forbid")

    query_id: str
    sample_idx: int = Field(ge=0, description="0..k-1.")
    seed: int = Field(description="Distinct within a (query_id, condition) group. C2.")
    condition: str = Field(description="'healthy' or 'F1'..'F5'.")
    model_id: str = Field(description="Resolved from config/models.yaml for the condition's role.")
    digest: str = Field(description="Backend version pin from GET /v1/models. C9.")
    digest_source: str = Field(
        default="backend",
        description="'backend' when the catalogue supplied a real digest, 'sentinel' when it "
        "did not. A missing pin must be VISIBLE, never forged into something "
        "that looks real.",
    )
    text: str = Field(description="May be empty. An empty completion is data, not an error.")
    finish_reason: str
    parse_ok: bool = Field(
        description="False for missing, empty, or unparseable completions. "
        "C6: these rows are NEVER dropped - parse failure is itself a "
        "degradation signal. Every signal reports the rate alongside its score."
    )
    submitted_at: datetime
    returned_at: datetime | None = None

    @property
    def group_key(self) -> tuple[str, str]:
        """The (query_id, condition) group whose k samples S3 compares."""
        return (self.query_id, self.condition)


class Window(BaseModel):
    """A slice-stratified window of queries and all their completions.

    The unit over which every threshold, the false-alarm budget, and detection
    latency are defined. Built only by `signals.windows` - never assemble one
    by hand, or the strata guarantee is lost and S5/S6 quietly degrade.
    """

    model_config = ConfigDict(extra="forbid")

    window_idx: int = Field(
        ge=0, description="Position in the replay stream; latency counts these."
    )
    query_ids: list[str] = Field(description="Exactly settings.window_size ids, stratified.")
    completions: list[Completion]
    condition: str = Field(description="Dominant condition; a post-onset window may be mixed.")

    @property
    def parse_fail_rate(self) -> float:
        """Fraction of this window's completions that failed to parse (C6)."""
        if not self.completions:
            return 0.0
        return sum(1 for c in self.completions if not c.parse_ok) / len(self.completions)


class SignalScore(BaseModel):
    """One signal's reading on one window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(description="One of settings.SIGNAL_NAMES.")
    value: float = Field(description="Higher = more degraded, for EVERY signal. See note below.")
    parse_fail_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Parse-failure rate of the rows this score was computed on (C6).",
    )
    n_rows: int = Field(ge=0, description="Rows contributing, INCLUDING parse failures.")

    # Orientation matters and is easy to get wrong: signals whose natural
    # reading falls under degradation (self-consistency, format validity) must
    # be inverted at source so that "higher = worse" holds uniformly. A
    # threshold comparison is always `value > threshold`; there is no
    # per-signal direction flag anywhere in detect/.


class Thresholds(BaseModel):
    """The frozen calibration record - `data/thresholds.json`.

    Written once by `detect.freeze.write()`. Never edited by hand, never
    recalibrated after freeze (C4). `evaluate/` refuses to run without a file
    that passes `freeze.verify()`.
    """

    model_config = ConfigDict(extra="forbid")

    frozen_at: datetime
    git_commit: str
    calib_rows_hash: str = Field(description="'sha256:...' over the exact calibration rows used.")
    k: int
    window_size: int
    window_stride: int
    family_wise_budget: float = Field(description="P(any signal fires) on a healthy window.")
    per_signal_alpha: float = Field(description="Sidak-corrected level actually used per signal.")
    n_signals: int = Field(description="m in the Sidak correction. Must equal len(thresholds).")
    bootstrap_windows: int
    calibration_seed: int
    refusal_patterns_version: str = Field(
        description="From refusal_patterns.yaml. If the patterns change after freeze, the frozen "
        "refusal threshold no longer means what it meant and evaluate/ must refuse to run."
    )
    prompts_version: str = Field(
        description="From prompts.yaml. Thresholds describe model behaviour on a "
        "specific set of prompts; if a template changes after freeze, the frozen "
        "numbers describe a different experiment and evaluate/ must refuse to run."
    )
    thresholds: dict[str, float]
    scoring: Literal["pooled", "per_slice"] = Field(
        default="pooled",
        description="How window scores were computed (signals.stratified). 'pooled' is v1 and "
        "the default, so records frozen before the field existed still read as what they were.",
    )

    @model_validator(mode="after")
    def _check_signal_count(self) -> Thresholds:
        if self.n_signals != len(self.thresholds):
            raise ValueError(
                f"n_signals={self.n_signals} but {len(self.thresholds)} thresholds present; "
                "the Sidak correction was computed for a different signal set"
            )
        return self

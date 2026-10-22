"""Central configuration. FULLY IMPLEMENTED - do not stub anything here.

Named `settings.py`, NOT `config.py`: a `config.py` module cannot sit beside
the `config/` package directory in the same parent (import collision, and it
will not lay down cleanly on a case-insensitive filesystem).

Everything backend-specific lives here or in `config/*.yaml`. No module in this
project may hardcode a base URL, an API key, a provider name, or a model id.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, ValidationInfo, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Package paths ──────────────────────────────────────────────────────────

PACKAGE_ROOT: Path = Path(__file__).parent
CONFIG_DIR: Path = PACKAGE_ROOT / "config"
PROJECT_ROOT: Path = PACKAGE_ROOT.parent
DATA_DIR: Path = PROJECT_ROOT / "data"

# ── Domain vocabulary ──────────────────────────────────────────────────────

Slice = Literal["gsm8k", "triviaqa", "squad2", "banking77", "xsum", "eli5", "jsonx"]
Split = Literal["calib", "heldout", "fault"]
Condition = Literal[
    "healthy",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "H2",
    "H3",
    "H4",
    "H5",
    "R",
    "R2",
    "U1",
    "Q1",
]

SLICES: tuple[str, ...] = ("gsm8k", "triviaqa", "squad2", "banking77", "xsum", "eli5", "jsonx")

#: Slices where `ground_truth` is non-null and the oracle can score correctness.
VERIFIABLE_SLICES: frozenset[str] = frozenset({"gsm8k", "triviaqa", "squad2", "banking77", "jsonx"})

#: Corpus composition. Sums to 5,000; the verifiable slices sum to exactly
#: 2,800, which is what the Stage 1 gate checks.
SLICE_SIZES: dict[str, int] = {
    "gsm8k": 700,
    "triviaqa": 700,
    "squad2": 700,
    "banking77": 500,
    "jsonx": 200,
    "xsum": 1100,
    "eli5": 1100,
}

#: Split proportions. Chosen so every slice divides with NO remainder - see
#: SPLIT_SIZES below. Do not change one without re-checking the other.
SPLIT_FRACTIONS: dict[str, float] = {"calib": 0.50, "heldout": 0.20, "fault": 0.30}

#: Exact per-slice, per-split counts. Stage 1's gate compares against this.
SPLIT_SIZES: dict[str, dict[str, int]] = {
    "gsm8k": {"calib": 350, "heldout": 140, "fault": 210},
    "triviaqa": {"calib": 350, "heldout": 140, "fault": 210},
    "squad2": {"calib": 350, "heldout": 140, "fault": 210},
    "banking77": {"calib": 250, "heldout": 100, "fault": 150},
    "jsonx": {"calib": 100, "heldout": 40, "fault": 60},
    "xsum": {"calib": 550, "heldout": 220, "fault": 330},
    "eli5": {"calib": 550, "heldout": 220, "fault": 330},
}

#: Slice composition of ONE window. Sums to WINDOW_SIZE (200) and is
#: proportional to SLICE_SIZES.
#:
#: Stratification is load-bearing, not cosmetic. Sampled without it, a window
#: would carry ~8 jsonx and ~20 banking77 rows only ON AVERAGE - many windows
#: would hold far fewer, and S5 (format validity) and S6 (label entropy) would
#: be computed on a handful of rows and become noise. Every window gets the
#: same composition so those two signals are comparable across windows.
WINDOW_STRATA: dict[str, int] = {
    "gsm8k": 28,
    "triviaqa": 28,
    "squad2": 28,
    "banking77": 20,
    "jsonx": 8,
    "xsum": 44,
    "eli5": 44,
}

#: The real-traffic study (Stage 7u, config/chat_study.yaml) has no task labels:
#: every question is one "chat" slice, so a window is 200 chat questions. Set
#: DRIFTWATCH_STUDY=chat for every command of that study; unset, the standard
#: strata above are untouched. Updated in place: every module holds this dict.
CHAT_SLICE: str = "chat"
if os.environ.get("DRIFTWATCH_STUDY") == "chat":
    WINDOW_STRATA.update(dict.fromkeys(WINDOW_STRATA, 0))
    WINDOW_STRATA[CHAT_SLICE] = 200

#: Slice mixture for F4 (query-distribution shift). Deliberately different from
#: SLICE_SIZES: long-form heavy, reasoning/structured light. F4 is a DERIVED
#: VIEW over `calib | fault`, not a fourth split - the `fault` split is fully
#: consumed by the shared F1/F2/F3 subset (C8) and `heldout` is ring-fenced by
#: C5, so there is nowhere else for F4's queries to come from. Prompt overlap
#: with calib is correct here: it isolates mixture change as the only variable.
F4_MIXTURE: dict[str, int] = {
    "gsm8k": 75,
    "triviaqa": 150,
    "squad2": 150,
    "banking77": 75,
    "jsonx": 0,
    "xsum": 525,
    "eli5": 525,
}

#: Slice mixture for F6, the OPPOSITE harmless shift: short-answer heavy,
#: long-form light. Added after the first evaluation, as an UNSEEN harmless
#: control: v2 and v3 scoring were both designed after looking at F4, so F4
#: can no longer judge them fairly. Fixed and committed before any scoring was
#: run on it; derived from healthy completions exactly like F4.
F6_MIXTURE: dict[str, int] = {
    "gsm8k": 400,
    "triviaqa": 400,
    "squad2": 350,
    "banking77": 200,
    "jsonx": 100,
    "xsum": 25,
    "eli5": 25,
}

#: Canonical signal names. The registry keys, the threshold keys in
#: thresholds.json, and the SignalScore.name values are all drawn from here.
SIGNAL_NAMES: tuple[str, ...] = (
    "length_ks",
    "refusal_rate",
    "self_consistency",
    "lexical_ttr",
    "format_validity",
    "label_entropy",
    "embedding_shift",
)

#: Batch endpoint row limits, read from the platform's batch validator
#: (backend/services/batch_validator.py). A submission exceeding either is
#: rejected wholesale, so submit.py must chunk against these.
MAX_ROWS_PER_BATCH: int = 50_000
MAX_BATCH_FILE_BYTES: int = 100 * 1024 * 1024

#: The only two batch URLs the platform accepts.
CHAT_URL: str = "/v1/chat/completions"
EMBEDDINGS_URL: str = "/v1/embeddings"


class Settings(BaseSettings):
    """Runtime configuration, sourced from environment or a local .env file.

    `base_url` and `api_key` have NO defaults on purpose: a default backend
    URL in the source tree is exactly what this project forbids.
    """

    model_config = SettingsConfigDict(env_prefix="DRIFTWATCH_", env_file=".env", extra="forbid")

    # ── Backend ────────────────────────────────────────────────────────────
    base_url: str = Field(description="OpenAI-compatible batch endpoint root, e.g. https://host/v1")
    api_key: str = Field(description="Bearer token for the batch endpoint.")
    request_timeout_s: float = 60.0

    # ── Remote API (OpenRouter-style router; optional) ─────────────────────
    # Used only by the remote-* commands (provider.remote). Like the backend
    # above they have no default URL in the source tree; leave both unset and
    # nothing remote can run.
    remote_base_url: str | None = Field(
        default=None, description="OpenAI-compatible router root, e.g. https://host/api/v1"
    )
    remote_api_key: str | None = Field(
        default=None, repr=False, description="Bearer token for the remote router."
    )
    poll_interval_s: float = 60.0
    poll_timeout_s: float = 24 * 3600.0

    # ── Sampling ───────────────────────────────────────────────────────────
    k: int = Field(
        default=4, ge=2, description="Samples per query. C1: k SEPARATE rows, never n>k."
    )
    temperature: float = 0.7
    max_tokens: int = 512
    seed_base: int = 20260901

    # ── Windows (see WINDOW_STRATA) ────────────────────────────────────────
    window_size: int = Field(default=200, description="Queries per window (W).")
    window_stride: int = Field(default=50, description="Fresh queries between window starts (S).")

    # ── Calibration ────────────────────────────────────────────────────────
    false_alarm_budget: float = Field(
        default=0.001,
        description="FAMILY-WISE budget: P(any signal fires on a healthy window). "
        "Per-signal levels are Sidak-corrected down from this - see "
        "detect.calibrate.sidak_alpha. Calibrating each signal AT this level "
        "would give a true family-wise rate of ~7x the budget.",
    )
    bootstrap_windows: int = Field(
        default=20_000,
        description="Resampled windows for threshold estimation. The calib split "
        "yields only ~46 real windows at W=200/S=50, and a 99.9th percentile "
        "cannot be read off 46 observations. Note this is still not enough for a "
        "raw empirical quantile at the Sidak-corrected level (~3 tail "
        "observations) - detect.calibrate falls back to a peaks-over-threshold "
        "tail fit, which is the correct method here. See its docstring.",
    )
    bootstrap_resamples: int = Field(
        default=2_000, ge=2_000, description="Resamples for evaluation CIs."
    )
    calibration_seed: int = 11071987
    n_workers: int = Field(
        default=0,
        description="CPU workers for the calibration bootstrap and evaluation "
        "resampling. 0 auto-detects (all cores but one); negative means 'all but "
        "N'. There is no device/GPU option: no model runs locally (C7), so the "
        "work is numpy/scipy and parallelises across cores, not accelerators. "
        "Changing this must NEVER change the numbers - see parallel.py.",
    )
    corpus_seed: int = 31415926

    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: Path = DATA_DIR

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("window_stride")
    @classmethod
    def _stride_fits(cls, v: int, info: ValidationInfo) -> int:
        w = info.data.get("window_size")
        if w is not None and v > w:
            raise ValueError(f"window_stride ({v}) must not exceed window_size ({w})")
        return v

    # ── Derived paths ──────────────────────────────────────────────────────
    @property
    def queries_path(self) -> Path:
        """Path to queries.jsonl."""
        return self.data_dir / "queries.jsonl"

    @property
    def completions_path(self) -> Path:
        """Path to completions.parquet."""
        return self.data_dir / "completions.parquet"

    @property
    def thresholds_path(self) -> Path:
        """Path to the frozen thresholds.json."""
        return self.data_dir / "thresholds.json"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide Settings singleton. Call `get_settings.cache_clear()` in tests."""
    return Settings()  # type: ignore[call-arg]


@lru_cache(maxsize=1)
def load_model_roles() -> dict[str, dict]:
    """Return the role -> catalogue-entry mapping from config/models.yaml.

    The ONLY sanctioned way to obtain a model id. Never inline one.
    """
    with (CONFIG_DIR / "models.yaml").open() as fh:
        return yaml.safe_load(fh)["roles"]


def model_id_for_role(role: str) -> str:
    """Resolve a role name ('healthy', 'weak', 'embedding', ...) to a model id."""
    roles = load_model_roles()
    if role not in roles:
        raise KeyError(f"Unknown role {role!r}. Known roles: {sorted(roles)}")
    return roles[role]["id"]


#: Which model role serves each experimental condition. F3 and F4 run on the
#: healthy model - they perturb the system prompt and the query mixture
#: respectively, not the weights.
CONDITION_ROLES: dict[str, str] = {
    "healthy": "healthy",
    "F1": "weak",
    "F2": "sibling",
    "F3": "healthy",
    "F4": "healthy",
    "F5": "cross_family",
    "F6": "healthy",
    "F7": "healthy",
    "F8": "healthy",
    "H2": "cross_family",
    "H3": "third_family",
    "H4": "fourth_family",
    "H5": "fifth_family",
    "R": "healthy",
    "R2": "healthy",
}

#: H2 is the SECOND HEALTHY MODEL (Stage 7n): gemma3:12b, the F5 model, on the
#: calib and heldout questions with the healthy prompt and seeds. Its answers to
#: the fault questions are F5's round, so together they mirror the healthy
#: round and the whole method can be re-run with gemma3 as the deployed model.
#: Never a fault: `evaluate` and the other replays leave it out.
SECOND_HEALTHY: str = "H2"
SECOND_HEALTHY_FAULT_ROUND: str = "F5"

#: H3 is the THIRD HEALTHY MODEL (Stage 7o): qwen2.5:14b, outside the Gemma
#: line altogether, on EVERY question with the healthy prompt and seeds - no
#: earlier round has its fault-question answers. Never a fault either.
THIRD_HEALTHY: str = "H3"
#: H4 is the FOURTH HEALTHY MODEL (Stage 7p): llama3.1:8b, a third family
#: outside the Gemma line, on every question like H3. Never a fault either.
FOURTH_HEALTHY: str = "H4"
#: H5 is the FIFTH HEALTHY MODEL (Stage 7q): phi4:14b, a fourth family outside
#: the Gemma line, on every question like H3 and H4. Never a fault either.
FIFTH_HEALTHY: str = "H5"
#: Every alternative deployed model; none of them is ever replayed as a fault.
ALTERNATE_HEALTHY: tuple[str, ...] = (
    SECOND_HEALTHY,
    THIRD_HEALTHY,
    FOURTH_HEALTHY,
    FIFTH_HEALTHY,
)

#: F7 is the healthy model rerun with FRESH seeds - a new day's sample of the
#: same deployment. Same model, prompt and questions; only this differs from
#: settings.seed_base, so no F7 answer repeats a healthy one by construction.
F7_SEED_BASE: int = 20261012

#: R is the paired monitor's REFERENCE round (Stage 7v, config/paired_study.yaml):
#: the healthy model with F7's seeds on the calib and heldout questions, so every
#: question has a fresh-seed healthy answer set. Written to its own data
#: directory, never to a study's store, and never replayed as a fault.
REFERENCE: str = "R"
#: R2 is the SECOND reference round (paired_study.yaml v1.1, design `rounds`):
#: the healthy model on the calib questions only, with seeds no other round
#: uses, so the calibration sees three healthy rounds (healthy, R, R2).
SECOND_REFERENCE: str = "R2"
R2_SEED_BASE: int = 20261022


@lru_cache(maxsize=1)
def load_refusal_patterns() -> dict:
    """Return {'version': str, 'patterns': list[str], 'empty_is_refusal': bool}."""
    with (CONFIG_DIR / "refusal_patterns.yaml").open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_prompts() -> dict:
    """Return {'version': str, 'system_prompts': {...}, 'slice_templates': {...}}.

    The ONLY sanctioned source of prompt text. Never inline a prompt in source:
    the calibrated thresholds describe model behaviour on THESE prompts, and an
    inline edit would silently redefine the experiment.
    """
    with (CONFIG_DIR / "prompts.yaml").open() as fh:
        return yaml.safe_load(fh)


def system_prompt_for_condition(condition: str) -> str:
    """System prompt for an experimental condition.

    F3 is the only condition that changes it - that substitution IS fault F3.
    Every other condition runs the healthy baseline prompt; F1/F2/F5 change the
    model and F4 changes the query mixture.
    """
    if condition == "F8":
        ext = load_extension_prompts()
        if not ext["f8_chosen"]:
            raise ValueError("F8 has no prompt yet: run the pilot and set f8_chosen first")
        return ext["f8_candidates"][ext["f8_chosen"]]
    prompts = load_prompts()["system_prompts"]
    key = "f3_corrupted" if condition == "F3" else "healthy"
    return prompts[key]


@lru_cache(maxsize=1)
def load_extension_prompts() -> dict:
    """Return config/extension_prompts.yaml: F8's candidate prompts and the chosen one.

    Kept apart from prompts.yaml so the version the frozen thresholds were
    calibrated on is untouched; see that file's header.
    """
    with (CONFIG_DIR / "extension_prompts.yaml").open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_judge_config(name: str = "judge") -> dict:
    """Return a pre-registered LLM-as-judge config: config/judge.yaml, or config/<name>.yaml."""
    with (CONFIG_DIR / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_api_study(name: str = "api_study") -> dict:
    """Return a pre-registered hosted study: api_study (Stage 7r) or chat_study (Stage 7u)."""
    with (CONFIG_DIR / f"{name}.yaml").open() as fh:
        return yaml.safe_load(fh)


@lru_cache(maxsize=1)
def load_remote_config() -> dict:
    """Return config/remote.yaml: the pinned remote deployments and the spending cap."""
    with (CONFIG_DIR / "remote.yaml").open() as fh:
        return yaml.safe_load(fh)


def slice_template(slice_name: str) -> str:
    """Prompt template for a corpus slice."""
    templates = load_prompts()["slice_templates"]
    if slice_name not in templates:
        raise KeyError(f"No prompt template for slice {slice_name!r}")
    return templates[slice_name]


@lru_cache(maxsize=1)
def load_jsonx_schema() -> dict:
    """Return the JSON Schema that S5 validates `jsonx` completions against."""
    with (CONFIG_DIR / "jsonx_schema.json").open() as fh:
        return json.load(fh)

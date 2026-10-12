"""Many unseen harmless shifts, not one (Stage 7d). Track T4.

v2 and v3 were designed after looking at F4, then judged on ONE unseen
control, F6. One mixture is an anecdote: it could be unusually easy or
unusually hard. This draws N_MIXTURES random query mixtures and replays each
exactly like F4 - healthy answers throughout, only WHAT is asked changes at
the onset - so every alarm after the onset is a false alarm. The result is a
distribution of false-alarm rates, and how it grows with the size of the shift.

The draw, fixed here before any real run:

  - slice shares p ~ Dirichlet(c * p_std), p_std the standard window mix
    (settings.WINDOW_STRATA) and c log-uniform on CONCENTRATION: small c gives
    large shifts, large c mild ones, so the set spans both (F4 and F6 sit at
    total-variation distance 0.26 and 0.41 from p_std);
  - MIXTURE_SIZE queries per mixture, from calib | fault - never heldout (C5) -
    with a slice's count capped at its pool and the excess handed to the others;
  - RUNS_PER_MIXTURE replays per mixture, F4's geometry.

Size of the shift: total-variation distance between the drawn mix and p_std.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from corpus.schemas import Completion, Query
from corpus.splits import fault_subset
from evaluate.metrics import signal_detection_rate, wilson_interval
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS
from evaluate.replay import ReplayRun, randomised_onsets, run_replay
from log import get_logger
from settings import WINDOW_STRATA

log = get_logger(__name__)

N_MIXTURES: int = 50
MIXTURE_SIZE: int = 1500
CONCENTRATION: tuple[float, float] = (4.0, 64.0)
RUNS_PER_MIXTURE: int = 100

#: The condition label every mixture's rows carry (one replay per mixture).
MIX: str = "MIX"

#: Total-variation bands for the summary table.
TV_BANDS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 1.0)

_STREAM: int = 8
_SLICES: tuple[str, ...] = tuple(WINDOW_STRATA)


@dataclass(frozen=True)
class Mixture:
    """One drawn mixture: its counts per slice, distance from standard, and queries."""

    idx: int
    concentration: float
    counts: dict[str, int]
    tv: float
    query_ids: tuple[str, ...]


def standard_shares() -> np.ndarray:
    """p_std, in WINDOW_STRATA order."""
    w = np.asarray([WINDOW_STRATA[s] for s in _SLICES], float)
    return w / w.sum()


def capped_counts(shares: np.ndarray, size: int, caps: np.ndarray) -> np.ndarray:
    """Integer counts summing to `size`, proportional to `shares`, none above its cap.

    Capped slices are fixed at their cap and the rest re-proportioned until
    nothing exceeds; then largest-remainder rounding, never past a cap.
    """
    if caps.sum() < size:
        raise ValueError(f"pool of {int(caps.sum())} cannot fill a mixture of {size}")
    target = np.zeros(len(shares))
    free = np.ones(len(shares), bool)
    left = float(size)
    while True:
        w = np.where(free, shares, 0.0)
        target[free] = left * w[free] / w[free].sum() if w[free].sum() else left / free.sum()
        over = free & (target > caps)
        if not over.any():
            break
        target[over] = caps[over]
        free &= ~over
        left = size - target[~free].sum()
    counts = np.floor(target).astype(int)
    for i in np.argsort(-(target - counts)):
        if counts.sum() == size:
            break
        if counts[i] < caps[i]:
            counts[i] += 1
    return counts


def draw_mixtures(queries: Sequence[Query], seed: int, n: int = N_MIXTURES) -> list[Mixture]:
    """The `n` mixtures, reproducible from `seed`."""
    pool: dict[str, list[str]] = {s: [] for s in _SLICES}
    for q in queries:
        if q.split in ("calib", "fault"):
            pool[q.slice].append(q.query_id)
    for s in pool:
        pool[s].sort()
    caps = np.asarray([len(pool[s]) for s in _SLICES])
    p_std = standard_shares()
    rng = np.random.default_rng(np.random.SeedSequence([seed, _STREAM]))
    lo, hi = np.log(CONCENTRATION[0]), np.log(CONCENTRATION[1])
    out = []
    for i in range(n):
        c = float(np.exp(rng.uniform(lo, hi)))
        counts = capped_counts(rng.dirichlet(c * p_std), MIXTURE_SIZE, caps)
        ids: list[str] = []
        for s, k in zip(_SLICES, counts, strict=True):
            picks = rng.choice(len(pool[s]), size=int(k), replace=False)
            ids += [pool[s][int(j)] for j in sorted(picks)]
        tv = 0.5 * float(np.abs(counts / MIXTURE_SIZE - p_std).sum())
        out.append(Mixture(i, c, dict(zip(_SLICES, map(int, counts), strict=True)), tv, tuple(ids)))
    return out


def run_mixtures(
    queries: Sequence[Query],
    healthy: Sequence[Completion],
    signals: dict,
    thresholds: dict[str, float],
    mixtures: Sequence[Mixture],
    *,
    n_runs: int = RUNS_PER_MIXTURE,
    seed: int,
) -> list[dict[str, Any]]:
    """One record per run (score traces dropped), tagged with its mixture.

    Each mixture's rows are healthy completions relabelled MIX; S7 needs the
    embedding store aliased under MIX (`pipeline.alias_embeddings`).
    """
    healthy = list(healthy)
    fault_queries = fault_subset(list(queries))
    by_query: dict[str, list[Completion]] = {}
    for c in healthy:
        by_query.setdefault(c.query_id, []).append(c)
    records: list[dict[str, Any]] = []
    for mix in mixtures:
        rows = [c.model_copy(update={"condition": MIX}) for q in mix.query_ids for c in by_query[q]]
        rng = np.random.default_rng(np.random.SeedSequence([seed, _STREAM, mix.idx]))
        onsets = randomised_onsets(
            n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, int(rng.integers(2**31))
        )
        run_seeds = rng.integers(0, 2**32, size=n_runs)
        runs = [
            ReplayRun(i, MIX, onset, DEFAULT_REPLAY_WINDOWS, int(run_seeds[i]))
            for i, onset in enumerate(onsets)
        ]
        for r in run_replay(runs, healthy, {MIX: rows}, fault_queries, signals, thresholds):
            r.pop("scores")
            r["mixture"] = mix.idx
            records.append(r)
        log.info("mixtures.done_one", mixture=mix.idx, tv=round(mix.tv, 3))
    return records


def summarise_mixtures(
    records: Sequence[dict[str, Any]], mixtures: Sequence[Mixture], names: Sequence[str]
) -> dict[str, Any]:
    """Per mixture, per TV band, and overall: how often a harmless shift alarms."""
    per_mixture = []
    for mix in mixtures:
        rs = [r for r in records if r["mixture"] == mix.idx]
        k = sum(r["latency"] is not None for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        lat = [r["latency"] for r in rs if r["latency"] is not None]
        per_mixture.append(
            {
                "mixture": mix.idx,
                "concentration": mix.concentration,
                "tv": mix.tv,
                "counts": mix.counts,
                "n_runs": len(rs),
                "alarmed": k,
                "rate": k / len(rs) if rs else float("nan"),
                "ci_lo": lo,
                "ci_hi": hi,
                "latency_median": float(np.median(lat)) if lat else None,
                "rate_by_signal": {n: signal_detection_rate(rs, n) for n in names},
            }
        )
    rates = np.asarray([m["rate"] for m in per_mixture])
    tvs = np.asarray([m["tv"] for m in per_mixture])
    bands = []
    for a, b in zip(TV_BANDS, TV_BANDS[1:], strict=False):
        inside = [m for m in per_mixture if a <= m["tv"] < b]
        if inside:
            runs = sum(m["n_runs"] for m in inside)
            k = sum(m["alarmed"] for m in inside)
            lo, hi = wilson_interval(k, runs)
            bands.append(
                {
                    "band": f"{a:.1f}-{b:.1f}",
                    "mixtures": len(inside),
                    "rate": k / runs,
                    "ci_lo": lo,
                    "ci_hi": hi,
                }
            )
    pre_w = sum(r["pre_onset_windows"] for r in records)
    pre_a = sum(r["pre_onset_alarms"] for r in records)
    rho = spearmanr(tvs, rates).statistic if len(set(rates.tolist())) > 1 else float("nan")
    return {
        "n_mixtures": len(per_mixture),
        "rate_mean": float(rates.mean()),
        "rate_median": float(np.median(rates)),
        "rate_q90": float(np.quantile(rates, 0.9)),
        "share_of_mixtures_over_half": float(np.mean(rates > 0.5)),
        "share_of_mixtures_under_5pct": float(np.mean(rates < 0.05)),
        "spearman_tv_vs_rate": float(rho),
        "by_band": bands,
        "rate_by_signal": {
            n: float(np.mean([m["rate_by_signal"][n] for m in per_mixture])) for n in names
        },
        "pre_onset_false_alarm_rate": pre_a / pre_w if pre_w else 0.0,
        "per_mixture": per_mixture,
    }


def pct(v: float) -> str:
    """A share as a percentage with one decimal."""
    return f"{100 * v:.1f}%"


def mixtures_markdown(summary: dict[str, Any], scoring: str) -> str:
    """Overall numbers, the TV-band table, and one row per mixture."""
    s = summary
    lines = [
        f"# {s['n_mixtures']} random harmless shifts ({scoring} scoring)\n",
        "Healthy answers throughout; only the query mix changes at the onset, so every alarm "
        "after it is a false alarm. Shift size: total-variation distance (TV) from the "
        "standard mix (F4 = 0.26, F6 = 0.41).\n",
        f"- mean alarm rate over mixtures: {pct(s['rate_mean'])} "
        f"(median {pct(s['rate_median'])}, 90th percentile {pct(s['rate_q90'])})",
        f"- mixtures alarming in more than half their runs: {pct(s['share_of_mixtures_over_half'])}",
        f"- mixtures alarming in under 5% of runs: {pct(s['share_of_mixtures_under_5pct'])}",
        f"- Spearman correlation, TV vs alarm rate: {s['spearman_tv_vs_rate']:.2f}",
        f"- pre-onset false-alarm rate: {s['pre_onset_false_alarm_rate']:.5f}",
        "- share of runs each signal alarmed in, averaged over mixtures: "
        + ", ".join(f"{n} {pct(v)}" for n, v in s["rate_by_signal"].items())
        + "\n",
        "## By size of shift\n",
        "| TV band | mixtures | alarm rate | 95% interval |",
        "|---|---|---|---|",
        *(
            f"| {b['band']} | {b['mixtures']} | {pct(b['rate'])} | "
            f"{100 * b['ci_lo']:.0f}-{100 * b['ci_hi']:.0f}% |"
            for b in s["by_band"]
        ),
        "\n## Every mixture\n",
        "| mixture | TV | " + " | ".join(_SLICES) + " | alarm rate | 95% interval |",
        "|---|---|" + "---|" * (len(_SLICES) + 2),
        *(
            f"| {m['mixture']} | {m['tv']:.3f} | "
            + " | ".join(str(m["counts"][x]) for x in _SLICES)
            + f" | {pct(m['rate'])} | {100 * m['ci_lo']:.0f}-{100 * m['ci_hi']:.0f}% |"
            for m in sorted(s["per_mixture"], key=lambda m: m["tv"])
        ),
    ]
    return "\n".join(lines) + "\n"

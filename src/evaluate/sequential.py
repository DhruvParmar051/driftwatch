"""CUSUM and Page-Hinkley next to the frozen thresholds (Stage 7j). Track T4.

The frozen monitor judges every window on its own. A sequential test adds up
small excursions over windows, so it can catch a drift that never crosses a
threshold in one window - the 5-10% partial faults are exactly that case.
`detect.detectors` has had both tests since the start; this calibrates and
evaluates them.

Design, fixed here before any real run:

  - Inputs: each signal's per-window score, standardised by its mean and
    standard deviation over healthy calibration streams.
  - Allowance: 0.5 healthy standard deviations, for both tests (the textbook
    choice for detecting a one-sd shift). Not tuned.
  - Decision level, per signal and test: the smallest level whose alarm rate
    per window on CALIB_STREAMS fully healthy streams of calib answers,
    replayed like the evaluation streams and reset after every alarm, is at
    most the frozen record's per-signal (Sidak) level. So all three detectors
    spend the same false-alarm budget, and every setting comes from healthy
    calib data only (C4).
  - Evaluation: the main replay streams exactly as `evaluate` builds them, and
    the partial/gradual streams exactly as `gradual` builds them. The
    threshold detector run here must reproduce the published numbers.
  - A detector alarms on a window when any signal does; detection, latency
    and false alarms before the onset are counted as everywhere else.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from corpus.schemas import Completion, Query
from evaluate.metrics import wilson_interval
from evaluate.pipeline import DEFAULT_MIN_PREFIX, DEFAULT_REPLAY_WINDOWS, replay_runs
from evaluate.replay import run_replay
from log import get_logger

log = get_logger(__name__)

#: Allowance, in healthy standard deviations, for CUSUM (k) and Page-Hinkley (delta).
ALLOWANCE: float = 0.5
CALIB_STREAMS: int = 2500
CALIB: str = "CALIB"
DETECTORS: tuple[str, ...] = ("threshold", "cusum", "page_hinkley")
_SEARCH_STEPS: int = 60


def calibration_traces(
    calib_queries: Sequence[Query],
    calib_rows: Sequence[Completion],
    signals: dict,
    thresholds: dict[str, float],
    *,
    n_streams: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Per signal, a [stream, window] matrix of scores on fully healthy calib streams.

    Built with the evaluation's own replay: the 'fault' is the same healthy
    calib answers under the name CALIB, so no window ever holds a faulty one.
    S7 needs those rows' vectors aliased under CALIB.
    """
    relabelled = [c.model_copy(update={"condition": CALIB}) for c in calib_rows]
    runs = replay_runs([CALIB], n_streams, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, seed)
    records = run_replay(
        runs, list(calib_rows), {CALIB: relabelled}, list(calib_queries), signals, thresholds
    )
    return traces(records, list(signals))


def traces(records: Sequence[dict[str, Any]], names: Sequence[str]) -> dict[str, np.ndarray]:
    """Stack the records' score traces into one [run, window] matrix per signal."""
    return {n: np.asarray([r["scores"][n] for r in records], dtype=float) for n in names}


def cusum_alarms(z: np.ndarray, k: float, h: float) -> np.ndarray:
    """One-sided upper CUSUM on every row of `z` at once, reset after each alarm.

    Same recursion as detect.detectors.CUSUMDetector.
    """
    s = np.zeros(z.shape[0])
    out = np.zeros(z.shape, dtype=bool)
    for t in range(z.shape[1]):
        s = np.maximum(0.0, s + z[:, t] - k)
        fire = s > h
        out[:, t] = fire
        s[fire] = 0.0
    return out


def page_hinkley_alarms(z: np.ndarray, delta: float, h: float) -> np.ndarray:
    """Page-Hinkley test for an increase, on every row at once, reset after each alarm.

    Same recursion as detect.detectors.PageHinkleyDetector.
    """
    rows = z.shape[0]
    n, mean, m, m_min = np.zeros(rows), np.zeros(rows), np.zeros(rows), np.zeros(rows)
    out = np.zeros(z.shape, dtype=bool)
    for t in range(z.shape[1]):
        n += 1
        mean += (z[:, t] - mean) / n
        m += z[:, t] - mean - delta
        m_min = np.minimum(m_min, m)
        fire = (m - m_min) > h
        out[:, t] = fire
        for arr in (n, mean, m, m_min):
            arr[fire] = 0.0
    return out


_TESTS = {"cusum": cusum_alarms, "page_hinkley": page_hinkley_alarms}


def decision_level(z: np.ndarray, test: str, alpha: float) -> float:
    """Smallest level whose per-window alarm rate on healthy `z` is at most `alpha`."""
    alarms = _TESTS[test]
    lo, hi = 0.0, 1.0
    while alarms(z, ALLOWANCE, hi).mean() > alpha:
        hi *= 2
        if hi > 1e6:
            raise RuntimeError(f"{test}: no level reaches alpha={alpha}")
    for _ in range(_SEARCH_STEPS):
        mid = (lo + hi) / 2
        if alarms(z, ALLOWANCE, mid).mean() > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def calibrate_sequential(calib: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    """Standardisation and decision levels, from healthy calib streams only."""
    scale = {n: (float(v.mean()), float(v.std())) for n, v in calib.items()}
    for n, (_mu, sd) in scale.items():
        if sd == 0:
            raise ValueError(f"{n} is constant on healthy calib streams; cannot standardise")
    z = {n: (v - scale[n][0]) / scale[n][1] for n, v in calib.items()}
    levels = {test: {n: decision_level(z[n], test, alpha) for n in calib} for test in _TESTS}
    realised = {
        test: float(
            np.mean(np.any([_TESTS[test](z[n], ALLOWANCE, levels[test][n]) for n in calib], axis=0))
        )
        for test in _TESTS
    }
    return {
        "allowance": ALLOWANCE,
        "per_signal_alpha": alpha,
        "scale": scale,
        "levels": levels,
        "calib_family_wise_rate": realised,
        "calib_windows": int(next(iter(calib.values())).size),
    }


def detector_alarms(
    scores: dict[str, np.ndarray], thresholds: dict[str, float], fitted: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Per detector, a [run, window] matrix: does any signal alarm on that window?"""
    out = {"threshold": np.any([scores[n] > thresholds[n] for n in scores], axis=0)}
    for test, alarms in _TESTS.items():
        per = []
        for n, v in scores.items():
            mu, sd = fitted["scale"][n]
            per.append(alarms((v - mu) / sd, ALLOWANCE, fitted["levels"][test][n]))
        out[test] = np.any(per, axis=0)
    return out


def summarise_detector(fired: np.ndarray, onsets: np.ndarray) -> dict[str, Any]:
    """Detection, latency and false alarms before the onset for one detector on one condition."""
    runs, windows = fired.shape
    after = np.arange(windows)[None, :] >= onsets[:, None]
    post = fired & after
    detected = post.any(axis=1)
    first = np.where(detected, post.argmax(axis=1), -1)
    latencies = (first - onsets)[detected]
    pre_windows = int(onsets.sum())
    lo, hi = wilson_interval(int(detected.sum()), runs)
    return {
        "n_runs": runs,
        "detected": int(detected.sum()),
        "detection_ci_lo": lo,
        "detection_ci_hi": hi,
        "latency_median": float(np.median(latencies)) if latencies.size else float("nan"),
        "pre_onset_false_alarm_rate": float((fired & ~after).sum() / pre_windows),
    }


def pre_onset_offset(records: Sequence[dict[str, Any]], fitted: dict[str, Any]) -> dict[str, float]:
    """Per signal, the mean standardised score of every healthy window before an onset.

    Zero would mean the healthy fault-subset answers look exactly like the calib
    streams the levels were set on. A positive offset is harmless to a
    one-window threshold but accumulates in CUSUM and Page-Hinkley.
    """
    out = {}
    for n, (mu, sd) in fitted["scale"].items():
        pre = [v for r in records for v in r["scores"][n][: r["onset"]]]
        out[n] = float((np.mean(pre) - mu) / sd)
    return out


def compare(
    records: Sequence[dict[str, Any]],
    thresholds: dict[str, float],
    fitted: dict[str, Any],
    key: Sequence[str],
) -> list[dict[str, Any]]:
    """One row per group of records (grouped by `key` fields) and detector."""
    names = list(thresholds)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault(tuple(r[k] for k in key), []).append(r)
    out = []
    for group, rs in sorted(groups.items()):
        fired = detector_alarms(traces(rs, names), thresholds, fitted)
        onsets = np.asarray([r["onset"] for r in rs])
        for det in DETECTORS:
            out.append(
                {**dict(zip(key, group, strict=True)), "detector": det}
                | summarise_detector(fired[det], onsets)
            )
    log.info("sequential.compared", groups=len(groups))
    return out


def sequential_markdown(
    fitted: dict[str, Any], main: Sequence[dict], gradual: Sequence[dict], scoring: str
) -> str:
    """Two tables, detectors side by side: whole-model faults, then partial and gradual ones."""

    def cell(r: dict) -> str:
        lat = "-" if np.isnan(r["latency_median"]) else f"{r['latency_median']:g}"
        return (
            f"{r['detected']}/{r['n_runs']} ({100 * r['detection_ci_lo']:.0f}-"
            f"{100 * r['detection_ci_hi']:.0f}%), {lat}"
        )

    def table(rows: Sequence[dict], key: Sequence[str]) -> list[str]:
        by: dict[tuple, dict[str, dict]] = {}
        for r in rows:
            by.setdefault(tuple(r[k] for k in key), {})[r["detector"]] = r
        head = " | ".join(key)
        lines = [
            f"| {head} | " + " | ".join(DETECTORS) + " | pre-onset false alarms (T / C / PH) |",
            "|" + "---|" * (len(key) + len(DETECTORS) + 1),
        ]
        for group, per in by.items():
            fa = " / ".join(f"{100 * per[d]['pre_onset_false_alarm_rate']:.2f}%" for d in DETECTORS)
            lines.append(
                "| "
                + " | ".join(map(str, group))
                + " | "
                + " | ".join(cell(per[d]) for d in DETECTORS)
                + f" | {fa} |"
            )
        return lines

    rates = fitted["calib_family_wise_rate"]
    lines = [
        f"# CUSUM and Page-Hinkley next to the frozen thresholds ({scoring} scoring)\n",
        f"Inputs standardised on healthy calib streams; allowance {fitted['allowance']} sd; "
        f"each signal's level set so its alarm rate per healthy calib window is at most "
        f"{fitted['per_signal_alpha']:.6f}, the frozen per-signal level "
        f"({fitted['calib_windows']} calib windows). Any-signal rate on those windows: "
        f"CUSUM {100 * rates['cusum']:.3f}%, Page-Hinkley {100 * rates['page_hinkley']:.3f}%.\n",
        "Healthy windows before an onset, in calib standard deviations (0 = like calib): "
        + ", ".join(f"{n} {v:+.2f}" for n, v in fitted.get("pre_onset_offset_sd", {}).items())
        + ".\n",
        "Each cell: runs detected (95% interval), median latency in windows.\n",
        "## Whole-model faults and controls (40-window streams, as `evaluate`)\n",
        *table(main, ["condition"]),
        "\n## Partial and gradual faults (60-window streams, as `gradual`)\n",
        *table(gradual, ["condition", "schedule"]),
    ]
    return "\n".join(lines) + "\n"

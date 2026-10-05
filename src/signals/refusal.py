"""S2 - refusal rate. Track T3."""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from corpus.schemas import Completion, SignalScore, Window
from settings import load_refusal_patterns
from signals.groups import GroupedSignal, GroupTable, require_reference


class RefusalRateSignal(GroupedSignal):
    """Excess rate of refusal-shaped completions over the healthy baseline.

    Targets: models that become over-cautious, and corrupted system prompts (F3).
    """

    name: str = "refusal_rate"

    def __init__(self) -> None:
        """Compile the patterns from config/refusal_patterns.yaml.

        Constraints:
            Patterns come from the versioned config file, NEVER inline. The
            file's `version` is recorded in thresholds.json: if the patterns
            change after a freeze, the frozen threshold no longer means what it
            meant and evaluate/ must refuse to run.
        """
        config = load_refusal_patterns()
        self.patterns_version: str = str(config["version"])
        self._patterns = [re.compile(p, re.IGNORECASE) for p in config["patterns"]]
        self._empty_is_refusal = bool(config.get("empty_is_refusal", False))
        self._baseline = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Record the baseline refusal rate on healthy calibration data.

        Constraints:
            The healthy baseline is not zero. A high-recall pattern set that
            fires on healthy text pushes the calibrated threshold up and
            silently costs detection power later - prefer precision.
        """
        require_reference(self.name, len(reference))
        self._baseline = sum(self.is_refusal(c) for c in reference) / len(reference)
        self._fitted = True

    def score(self, window: Window) -> SignalScore:
        """Refusal rate in the window, relative to baseline.

        Behaviour:
            Match each completion's FULL text against the compiled patterns
            (case-insensitive). `value` is the excess over the fitted baseline,
            clipped at 0 - a window refusing LESS than baseline is not degraded.

        Edge cases:
            - An empty completion is NOT a refusal. It is a parse failure (C6),
              counted in parse_fail_rate. Conflating the two makes S2 fire on
              truncation, which is S1's job, and the two faults become
              indistinguishable. `empty_is_refusal` in the config governs this
              and defaults to false.
        """
        return super().score(window)

    def is_refusal(self, c: Completion) -> bool:
        """Whether one completion's full text matches a refusal pattern.

        Empty text is a parse failure, not a refusal, unless the config says
        otherwise (it does not, by default).
        """
        if not c.text.strip():
            return self._empty_is_refusal
        return any(p.search(c.text) for p in self._patterns)

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group refusal, row and parse-failure counts."""
        return {
            "refusals": np.array([sum(self.is_refusal(c) for c in g) for g in groups], float),
            "rows": np.array([len(g) for g in groups], float),
            "fails": np.array([sum(not c.parse_ok for c in g) for g in groups], float),
        }

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Refusal rate minus the healthy baseline, clipped at 0."""
        self._require_fitted()
        rows = table["rows"][idx].sum()
        fails = table["fails"][idx].sum()
        if rows == 0:
            return self._result(0.0, 0, 0)
        rate = table["refusals"][idx].sum() / rows
        return self._result(max(0.0, rate - self._baseline), rows, fails)

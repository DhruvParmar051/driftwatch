"""S2 - refusal rate. STUB - track T3."""

from __future__ import annotations

from corpus.schemas import Completion, SignalScore, Window


class RefusalRateSignal:
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
        raise NotImplementedError

    def fit(self, reference: list[Completion]) -> None:
        """Record the baseline refusal rate on healthy calibration data.

        Constraints:
            The healthy baseline is not zero. A high-recall pattern set that
            fires on healthy text pushes the calibrated threshold up and
            silently costs detection power later - prefer precision.
        """
        raise NotImplementedError

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
        raise NotImplementedError

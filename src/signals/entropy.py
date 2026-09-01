"""S6 - Shannon entropy over the banking77 label distribution. STUB - track T3."""

from __future__ import annotations

from corpus.schemas import Completion, SignalScore, Window


class LabelEntropySignal:
    """Shift in the entropy of predicted intent labels vs the healthy reference.

    Targets: mode collapse and label drift. A degraded classifier stops using
    the full label space and collapses onto a few high-frequency intents, which
    drops entropy sharply - while per-item accuracy may look only mildly worse.
    """

    name: str = "label_entropy"

    def __init__(self) -> None:
        raise NotImplementedError

    def fit(self, reference: list[Completion]) -> None:
        """Record the baseline label distribution and its entropy.

        Constraints:
            Only banking77 rows participate: WINDOW_STRATA['banking77'] = 20
            queries x k = 80 rows per window, over a 77-label space. That is
            fewer rows than labels, so the per-window empirical distribution is
            sparse by construction. Use a bias-corrected estimator
            (Miller-Madow or similar) or the entropy is systematically
            underestimated in EVERY window, including healthy ones - which
            biases the baseline, not just the fault windows.
        """
        raise NotImplementedError

    def score(self, window: Window) -> SignalScore:
        """Absolute entropy shift from baseline, oriented higher = worse.

        Behaviour:
            Map each banking77 completion to a label in the query's
            `label_space`, build the empirical distribution, compute entropy,
            and return |entropy - baseline|.

        Constraints:
            Direction: BOTH directions are degradation. Collapse onto few labels
            drops entropy; random guessing raises it. Use the absolute shift, not
            a signed one, or half the failure modes are invisible.

            Label mapping is exact-match against label_space, with unmapped
            completions counted as parse failures (C6) - NOT silently assigned
            to a nearest label. Fuzzy mapping would mask exactly the
            instruction-following decay this signal exists to detect.
        """
        raise NotImplementedError

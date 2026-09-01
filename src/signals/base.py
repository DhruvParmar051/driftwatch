"""The Signal protocol. FULLY IMPLEMENTED - the contract track T3 implements against.

Every signal is a class with `fit(reference)` then `score(window)`. Adding a
signal means: implement this protocol, register it in `registry.py`, add its
name to `settings.SIGNAL_NAMES`, and add a flat/monotone test. Nothing else.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from corpus.schemas import Completion, SignalScore, Window


class NotFittedError(RuntimeError):
    """Raised when `score()` is called before `fit()`."""


class InsufficientReferenceError(ValueError):
    """Raised when the reference sample is too small for a stable statistic.

    Raise this rather than returning a noisy number. A signal fitted on too
    little data produces a threshold that is quietly wrong, and nothing
    downstream can detect that it happened.
    """


@runtime_checkable
class Signal(Protocol):
    """A label-free degradation signal.

    Implementations MUST NOT import anything from the `oracle` package - the
    signals are label-free by definition and an accidental import invalidates
    the entire result (C3, enforced by tests/test_no_oracle_import.py).
    """

    #: Must be one of settings.SIGNAL_NAMES.
    name: str

    def fit(self, reference: list[Completion]) -> None:
        """Fit the reference distribution from HEALTHY CALIBRATION rows only.

        Args:
            reference: healthy completions drawn exclusively from split == "calib".
                Callers guarantee this; `detect.calibrate` raises otherwise (C4).

        Raises:
            InsufficientReferenceError: if the reference is too small to fit.

        Constraints:
            C6 - rows with parse_ok=False are part of the reference, not filtered
                 out. The healthy parse-failure rate is a real property of the
                 baseline and excluding it biases every later comparison.
        """
        ...

    def score(self, window: Window) -> SignalScore:
        """Score one window against the fitted reference.

        Returns:
            SignalScore whose `value` is oriented HIGHER = MORE DEGRADED. Signals
            whose natural reading falls under degradation (self-consistency,
            format validity) must be inverted here, at source. detect/ compares
            with a bare `value > threshold` and carries no direction flag.

        Raises:
            NotFittedError: if `fit()` has not been called.
        """
        ...

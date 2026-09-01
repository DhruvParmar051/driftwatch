"""Signal registry. FULLY IMPLEMENTED.

Single source of truth for which signals exist and in what order. Import
`build_signals()` rather than instantiating signal classes directly, so that
enabling or disabling a signal is a one-line change here.
"""

from __future__ import annotations

from driftwatch.settings import SIGNAL_NAMES
from driftwatch.signals.base import Signal
from driftwatch.signals.consistency import SelfConsistencySignal
from driftwatch.signals.embedding import EmbeddingShiftSignal
from driftwatch.signals.entropy import LabelEntropySignal
from driftwatch.signals.fmt import FormatValiditySignal
from driftwatch.signals.length import LengthKSSignal
from driftwatch.signals.lexical import LexicalDiversitySignal
from driftwatch.signals.refusal import RefusalRateSignal

#: name -> constructor. Order is fixed and matches settings.SIGNAL_NAMES.
SIGNAL_CLASSES: dict[str, type] = {
    "length_ks": LengthKSSignal,
    "refusal_rate": RefusalRateSignal,
    "self_consistency": SelfConsistencySignal,
    "lexical_ttr": LexicalDiversitySignal,
    "format_validity": FormatValiditySignal,
    "label_entropy": LabelEntropySignal,
    "embedding_shift": EmbeddingShiftSignal,
}

assert tuple(SIGNAL_CLASSES) == SIGNAL_NAMES, "registry order must match settings.SIGNAL_NAMES"


def build_signals(enabled: list[str] | None = None) -> dict[str, Signal]:
    """Instantiate the signal set.

    Args:
        enabled: names to build, or None for all of them. The COUNT of the
            result is `m` in the Sidak correction, so disabling a signal
            changes every per-signal threshold. Resolve the set BEFORE
            calibration, never after.

    Raises:
        KeyError: on an unknown signal name.
    """
    names = list(SIGNAL_CLASSES) if enabled is None else list(enabled)
    unknown = set(names) - set(SIGNAL_CLASSES)
    if unknown:
        raise KeyError(f"Unknown signal(s): {sorted(unknown)}. Known: {sorted(SIGNAL_CLASSES)}")
    return {n: SIGNAL_CLASSES[n]() for n in names}

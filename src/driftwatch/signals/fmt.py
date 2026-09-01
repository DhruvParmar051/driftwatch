"""S5 - JSON schema validity on the `jsonx` slice. STUB - track T3."""

from __future__ import annotations

from driftwatch.corpus.schemas import Completion, SignalScore, Window


class FormatValiditySignal:
    """Rate of jsonx completions failing config/jsonx_schema.json.

    Targets: loss of instruction-following and structured-output ability, often
    the first thing to go when a model is swapped for a weaker one.
    """

    name: str = "format_validity"

    def __init__(self) -> None:
        """Load and compile the schema via settings.load_jsonx_schema()."""
        raise NotImplementedError

    def fit(self, reference: list[Completion]) -> None:
        """Record the baseline validity rate on healthy jsonx completions.

        Constraints:
            Only jsonx rows participate. Every window contains exactly
            WINDOW_STRATA['jsonx'] = 8 queries x k samples = 32 rows, which is
            small - it is why windows are stratified at all, and it means this
            signal is inherently noisier than the others. Expect a wider
            calibrated threshold and say so in the report.
        """
        raise NotImplementedError

    def score(self, window: Window) -> SignalScore:
        """Schema-violation rate on the window's jsonx rows, higher = worse.

        Behaviour:
            Validate each jsonx completion CLIENT-SIDE against the schema:
            parse the text as JSON, then validate the object. Count both parse
            failures and schema violations as invalid.

        Constraints:
            This validation MUST happen here. jsonx rows are submitted with no
            `response_format` precisely so that malformed output comes back as
            text - if strict json_schema mode were used, the platform would
            validate server-side and fail the row, and this signal would have
            nothing left to measure because the platform would be doing its job
            for it. If someone "fixes" expand.py to request strict mode, this
            signal silently goes to zero. Guard it with a test.

        Edge cases:
            - Models often wrap JSON in ``` fences or prose. Decide whether to
              strip fences before parsing and document it: stripping measures
              schema conformance, not stripping measures instruction-following.
              They are different signals and the choice changes what S5 means.
        """
        raise NotImplementedError

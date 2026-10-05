"""S5 - JSON schema validity on the `jsonx` slice. Track T3."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from jsonschema import Draft202012Validator

from corpus.schemas import Completion, SignalScore, Window
from settings import load_jsonx_schema
from signals.groups import GroupedSignal, GroupTable, require_reference
from textnorm import parse_json_object, slice_of

#: The only slice S5 reads.
JSONX: str = "jsonx"


class FormatValiditySignal(GroupedSignal):
    """Rate of jsonx completions failing config/jsonx_schema.json.

    Targets: loss of instruction-following and structured-output ability, often
    the first thing to go when a model is swapped for a weaker one.
    """

    name: str = "format_validity"

    def __init__(self) -> None:
        """Load and compile the schema via settings.load_jsonx_schema()."""
        self._validator = Draft202012Validator(load_jsonx_schema())
        self._baseline = 0.0
        self._fitted = False

    def fit(self, reference: list[Completion]) -> None:
        """Record the baseline validity rate on healthy jsonx completions.

        Constraints:
            Only jsonx rows participate. Every window contains exactly
            WINDOW_STRATA['jsonx'] = 8 queries x k samples = 32 rows, which is
            small - it is why windows are stratified at all, and it means this
            signal is inherently noisier than the others. Expect a wider
            calibrated threshold and say so in the report.
        """
        rows = [c for c in reference if slice_of(c.query_id) == JSONX]
        require_reference(self.name, len(rows), "jsonx rows")
        self._baseline = sum(not self.is_valid(c) for c in rows) / len(rows)
        self._fitted = True

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
        return super().score(window)

    def is_valid(self, c: Completion) -> bool:
        """Parses as JSON AND validates against config/jsonx_schema.json.

        Fence decision: ONE surrounding ``` / ```json fence is stripped before
        parsing (textnorm.strip_code_fences). S5 therefore measures SCHEMA
        CONFORMANCE, not fence avoidance; prose around the JSON still fails.
        """
        if not c.parse_ok:
            return False
        obj = parse_json_object(c.text)
        return obj is not None and self._validator.is_valid(obj)

    def prepare(self, groups: Sequence[Sequence[Completion]]) -> GroupTable:
        """Per-group invalid, jsonx-row and parse-failure counts (jsonx rows only)."""
        invalid, rows, fails = (np.zeros(len(groups)) for _ in range(3))
        for i, group in enumerate(groups):
            for c in group:
                if slice_of(c.query_id) != JSONX:
                    continue
                rows[i] += 1
                fails[i] += not c.parse_ok
                invalid[i] += not self.is_valid(c)
        return {"invalid": invalid, "rows": rows, "fails": fails}

    def score_indices(self, table: GroupTable, idx: np.ndarray) -> SignalScore:
        """Invalid rate minus the healthy baseline, clipped at 0.

        A window with no jsonx rows (the F4 mixture has none) carries no
        evidence either way and scores 0.0 with n_rows=0.
        """
        self._require_fitted()
        rows = table["rows"][idx].sum()
        if rows == 0:
            return self._result(0.0, 0, 0)
        rate = table["invalid"][idx].sum() / rows
        return self._result(max(0.0, rate - self._baseline), rows, table["fails"][idx].sum())

"""Answer normalisation shared by S3 and the oracle. NEUTRAL MODULE.

Both `signals.consistency` (S3) and `oracle.score` need to decide whether two
answers are "the same". C3 forbids anything reachable from detect/ - which
includes signals/ - from importing oracle/, so the shared logic lives here,
in a module that imports neither.

This module must never import `oracle`, `signals` or `detect`. It is pure
string handling with no knowledge of ground truth: S3 compares samples to EACH
OTHER with it, and only the oracle compares a sample to a ground truth.
"""

from __future__ import annotations

import json
import re
import string
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from typing import Any

import yaml

from settings import CONFIG_DIR

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)
_WORD = re.compile(r"\w+", re.UNICODE)
_NUMBER = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")
_FENCE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)
_LABEL_TRIM = " \t\r\n\"'`.:;"


def normalise_text(text: str) -> str:
    """SQuAD-style normalisation: lowercase, drop punctuation and articles, squeeze spaces."""
    text = text.lower().translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def word_tokens(text: str) -> list[str]:
    r"""Lowercased word tokens (`\w+`). The token definition for S3, S4 and S7."""
    return _WORD.findall(text.lower())


def extract_final_number(text: str) -> str | None:
    """The final numeric answer of a reasoning completion, canonicalised.

    Looks at the last non-empty line that contains a number and takes the last
    number on it - the gsm8k template asks for the answer on its own last line,
    and models that comply put it there. Commas and currency signs are
    stripped and the value is printed without trailing zeros ("18.00" -> "18").

    Returns None when the text contains no number at all.
    """
    for line in reversed(text.strip().splitlines()):
        matches = _NUMBER.findall(line)
        if matches:
            return _canonical_number(matches[-1])
    return None


def _canonical_number(raw: str) -> str | None:
    cleaned = raw.replace(",", "").replace("$", "")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(value.quantize(Decimal(1)))
    return format(value.normalize(), "f")


def strip_code_fences(text: str) -> str:
    """Remove one surrounding ``` / ```json fence if present.

    S5 strips fences before parsing, so it measures SCHEMA conformance rather
    than fence-avoidance. A model that wraps valid JSON in a fence has still
    produced the object it was asked for.
    """
    match = _FENCE.match(text)
    return match.group(1) if match else text


def parse_json_object(text: str) -> Any | None:
    """Parse a completion as JSON after stripping fences. None if it does not parse."""
    candidate = strip_code_fences(text).strip()
    if not candidate:
        return None
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, RecursionError):
        return None


@lru_cache(maxsize=1)
def _banking77_config() -> dict[str, Any]:
    with (CONFIG_DIR / "banking77_labels.yaml").open() as fh:
        return yaml.safe_load(fh)


def load_banking77_labels() -> list[str]:
    """The 77 intent labels from config/banking77_labels.yaml, in ClassLabel order."""
    return list(_banking77_config()["labels"])


def banking77_labels_version() -> str:
    """Version string of the label file."""
    return str(_banking77_config()["version"])


def map_label(text: str, labels: list[str]) -> str | None:
    """Map a completion to a label by EXACT match, or None.

    Only trivial wrapping is forgiven: surrounding whitespace, quotes,
    backticks, a trailing period, and letter case. No fuzzy or nearest-label
    matching - S6 exists to detect instruction-following decay, and a fuzzy map
    would repair exactly the failures it should be counting.
    """
    candidate = text.strip(_LABEL_TRIM).lower()
    if not candidate:
        return None
    for label in labels:
        if candidate == label.lower():
            return label
    return None


def canonical_answer(text: str, slice_name: str) -> str:
    """Canonical form of a completion for exact-match comparison on a slice.

    - gsm8k:     the final number, or "" if there is none;
    - banking77: the mapped label, or the normalised text if unmapped;
    - jsonx:     the parsed object re-serialised with sorted keys, or the
                 normalised text if it does not parse;
    - otherwise: SQuAD-style normalised text.

    Two completions are "the same answer" iff their canonical forms are equal.
    """
    if slice_name == "gsm8k":
        return extract_final_number(text) or ""
    if slice_name == "banking77":
        label = map_label(text, load_banking77_labels())
        return label if label is not None else normalise_text(text)
    if slice_name == "jsonx":
        obj = parse_json_object(text)
        if obj is not None:
            return json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return normalise_text(text)
    return normalise_text(text)


def slice_of(query_id: str) -> str:
    """Slice name from a query id ('gsm8k_00042' -> 'gsm8k')."""
    head, sep, _ = query_id.rpartition("_")
    return head if sep else query_id

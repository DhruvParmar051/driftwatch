"""SPEC - track T2, provider layer (Stage 2/3 gates).

Everything here runs against a mocked httpx transport. No test in this file may
touch a network - the whole provider layer is designed so that it does not have
to, because a real round costs a day.
"""

from __future__ import annotations

import json

import pytest

pytestmark = pytest.mark.spec


# ── custom_id encoding (V6: the row schema has nowhere else for metadata) ──


def test_custom_id_round_trips():
    from provider.expand import decode_custom_id, encode_custom_id

    cid = encode_custom_id("gsm8k_00042", 3, "F2", "r007")
    assert decode_custom_id(cid) == ("gsm8k_00042", 3, "F2", "r007")


def test_custom_id_rejects_separator_in_components():
    from provider.expand import CUSTOM_ID_SEP, encode_custom_id

    with pytest.raises(ValueError):
        encode_custom_id(f"bad{CUSTOM_ID_SEP}id", 0, "healthy", "r001")


def test_malformed_custom_id_raises_rather_than_returning_a_sentinel():
    """An unmatchable row is a reconciliation failure, not a row to discard."""
    from provider.expand import decode_custom_id

    with pytest.raises(ValueError):
        decode_custom_id("not-a-valid-custom-id")


# ── seeds (C2) ─────────────────────────────────────────────────────────────


def test_derived_seeds_are_distinct_within_a_query():
    from provider.expand import derive_seed

    seeds = {derive_seed("gsm8k_00042", i, 42) for i in range(4)}
    assert len(seeds) == 4


def test_derived_seeds_are_stable_across_processes():
    """Python's salted hash() is not acceptable here; hashlib is."""
    import subprocess
    import sys

    code = "from provider.expand import derive_seed;print(derive_seed('gsm8k_00042', 0, 42))"
    outputs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        # Without this the test passes vacuously while unimplemented: both runs
        # fail identically, stdout is empty twice, and the set has one element.
        assert proc.returncode == 0, proc.stderr
        outputs.append(int(proc.stdout.strip()))
    assert outputs[0] == outputs[1]


# ── row construction (C1, V7, V8) ──────────────────────────────────────────


@pytest.fixture
def chat_row(sample_query, settings):
    from provider.expand import build_chat_row

    return build_chat_row(
        sample_query,
        0,
        "test-model",
        "healthy",
        "r001",
        None,
        settings.seed_base,
        settings.temperature,
        settings.max_tokens,
    )


def test_row_has_only_the_four_allowed_top_level_keys(chat_row):
    """The platform rejects the ENTIRE FILE on an unknown top-level key."""
    assert set(chat_row) == {"custom_id", "method", "url", "body"}


def test_row_method_and_url_are_valid(chat_row):
    from settings import CHAT_URL

    assert chat_row["method"] == "POST"
    assert chat_row["url"] == CHAT_URL


def test_row_never_sets_n(chat_row):
    """C1 - the submission validator does NOT check `n`. It fails at the worker,
    18 hours later, as scattered failed rows. One mistake costs a full round.
    """
    assert "n" not in chat_row["body"]


def test_row_never_sets_stream(chat_row):
    assert "stream" not in chat_row["body"]


def test_row_carries_a_seed(chat_row):
    assert isinstance(chat_row["body"].get("seed"), int)


def test_jsonx_rows_do_not_request_strict_response_format(settings):
    """V8 - strict json_schema mode makes the PLATFORM validate and fail the row,
    leaving S5 with nothing to measure. Malformed JSON must come back as text.
    """
    from corpus.schemas import Query
    from provider.expand import build_chat_row

    q = Query(
        query_id="jsonx_00001",
        slice="jsonx",
        prompt="Return a JSON object...",
        ground_truth='{"title":"x"}',
        split="calib",
    )
    row = build_chat_row(
        q,
        0,
        "test-model",
        "healthy",
        "r001",
        None,
        settings.seed_base,
        settings.temperature,
        settings.max_tokens,
    )
    fmt = row["body"].get("response_format")
    assert fmt is None or fmt.get("type") != "json_schema"


def test_expand_produces_k_rows_per_query_with_distinct_seeds(sample_query, settings):
    from provider.expand import expand_queries

    rows = expand_queries(
        [sample_query],
        4,
        "test-model",
        "healthy",
        "r001",
        None,
        settings.seed_base,
        settings.temperature,
        settings.max_tokens,
    )
    assert len(rows) == 4
    assert len({r["body"]["seed"] for r in rows}) == 4
    assert all("n" not in r["body"] for r in rows)


def test_expand_refuses_to_exceed_the_platform_row_cap(settings):
    from corpus.schemas import Query
    from provider.expand import expand_queries
    from settings import MAX_ROWS_PER_BATCH

    n_queries = MAX_ROWS_PER_BATCH // settings.k + 1
    queries = [
        Query(query_id=f"xsum_{i:05d}", slice="xsum", prompt="p", split="calib")
        for i in range(n_queries)
    ]
    with pytest.raises(ValueError):
        expand_queries(
            queries,
            settings.k,
            "test-model",
            "healthy",
            "r001",
            None,
            settings.seed_base,
            settings.temperature,
            settings.max_tokens,
        )


# ── reconciliation (C10) ───────────────────────────────────────────────────


def test_missing_row_fails_loudly():
    """C10 - a partial batch breaks every per-cell sample size, invisibly."""
    from provider.collect import ReconciliationError, ReconciliationReport

    report = ReconciliationReport(n_submitted=100, n_returned=99, missing=["q|0|healthy|r1"])
    assert not report.is_clean
    with pytest.raises(ReconciliationError):
        report.raise_if_dirty()


def test_equal_counts_with_a_swap_still_fails():
    """A length check would pass this. Reconciliation is a SET comparison."""
    from provider.collect import ReconciliationError, ReconciliationReport

    report = ReconciliationReport(
        n_submitted=100, n_returned=100, missing=["a|0|healthy|r1"], duplicated=["b|0|healthy|r1"]
    )
    with pytest.raises(ReconciliationError):
        report.raise_if_dirty()


def test_clean_report_does_not_raise():
    from provider.collect import ReconciliationReport

    ReconciliationReport(n_submitted=100, n_returned=100).raise_if_dirty()


# ── parsing (C6) ───────────────────────────────────────────────────────────


def test_error_response_becomes_a_parse_failure_not_a_dropped_row():
    from provider.collect import parse_result_line

    line = json.dumps(
        {"custom_id": "q|0|healthy|r1", "response": None, "error": {"message": "worker failed"}}
    )
    custom_id, text, _finish, parse_ok = parse_result_line(line)
    assert custom_id == "q|0|healthy|r1"
    assert parse_ok is False and text == ""


def test_empty_completion_is_a_parse_failure():
    from provider.collect import parse_result_line

    line = json.dumps(
        {
            "custom_id": "q|0|healthy|r1",
            "response": {
                "body": {"choices": [{"message": {"content": "   "}, "finish_reason": "stop"}]}
            },
        }
    )
    _cid, text, _finish, parse_ok = parse_result_line(line)
    assert parse_ok is False and text.strip() == ""


def test_good_completion_parses():
    from provider.collect import parse_result_line

    line = json.dumps(
        {
            "custom_id": "q|0|healthy|r1",
            "response": {
                "body": {"choices": [{"message": {"content": "8 apples"}, "finish_reason": "stop"}]}
            },
        }
    )
    _cid, text, finish, parse_ok = parse_result_line(line)
    assert parse_ok is True and text == "8 apples" and finish == "stop"


# ── catalogue (C9) ─────────────────────────────────────────────────────────


def test_unresolvable_role_raises_rather_than_substituting():
    """A silent substitution makes the whole comparison meaningless."""
    import httpx

    from provider.catalogue import RoleResolutionError, resolve_roles
    from provider.client import BatchClient

    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"data": [{"id": "some-other-model"}]})
    )
    client = BatchClient("http://test.invalid/v1", "k", transport=transport)
    with pytest.raises(RoleResolutionError):
        resolve_roles(client)


def test_resolved_roles_carry_a_digest():
    """C9 - model drift between rounds is otherwise undetectable."""
    import httpx

    from provider.catalogue import resolve_roles
    from provider.client import BatchClient
    from settings import load_model_roles

    data = [{"id": e["id"], "digest": "d" * 64} for e in load_model_roles().values()]
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": data}))
    client = BatchClient("http://test.invalid/v1", "k", transport=transport)
    for resolved in resolve_roles(client).values():
        assert resolved.digest and resolved.digest_source == "backend"

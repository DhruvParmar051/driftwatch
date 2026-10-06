"""SPEC - the local Ollama path, offline.

A laptop round runs for hours and will be interrupted. These pin the failure
modes: a server that dies mid-round, a model that is not pulled, a request the
server rejects, a crash that tears the last checkpoint line, and a rerun that
must not duplicate a single row.
"""

from __future__ import annotations

import json

import pytest

from tests.fake_ollama import FakeOllama

pytestmark = pytest.mark.spec


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch) -> None:
    import provider.local

    monkeypatch.setattr(provider.local, "_BACKOFF_S", 0.0)


@pytest.fixture
def rows(settings):
    from corpus.schemas import Query
    from provider.expand import expand_queries
    from settings import model_id_for_role

    queries = [
        Query(query_id=f"eli5_{i:05d}", slice="eli5", prompt=f"why {i}?", split="calib")
        for i in range(5)
    ]
    return expand_queries(
        queries,
        settings.k,
        model_id_for_role("healthy"),
        "healthy",
        "r001",
        None,
        settings.seed_base,
        settings.temperature,
        settings.max_tokens,
    )


def client(fake):
    from provider.local import LocalClient

    return LocalClient("http://localhost:11434/v1", transport=fake.transport)


def ids_in(path):
    return [json.loads(line)["custom_id"] for line in path.read_text().splitlines()]


def test_every_row_is_answered_and_checkpointed(tmp_path, rows):
    from provider.local import run_rows

    ck = tmp_path / "results.jsonl"
    assert run_rows(client(FakeOllama()), rows, ck, concurrency=3) == len(rows)
    assert sorted(ids_in(ck)) == sorted(r["custom_id"] for r in rows)


def test_rerun_after_completion_sends_nothing(tmp_path, rows):
    from provider.local import run_rows

    fake, ck = FakeOllama(), tmp_path / "results.jsonl"
    run_rows(client(fake), rows, ck)
    sent = fake.served
    assert run_rows(client(fake), rows, ck) == 0
    assert fake.served == sent


def test_server_dying_midround_keeps_answers_and_resumes_without_duplicates(tmp_path, rows):
    """A dead server is NOT a model failure: those rows are retried, never recorded."""
    from provider.local import LocalServerError, run_rows

    ck = tmp_path / "results.jsonl"
    with pytest.raises(LocalServerError):
        run_rows(client(FakeOllama(down_after=7)), rows, ck, concurrency=1)
    kept = ids_in(ck)
    assert 0 < len(kept) < len(rows)

    run_rows(client(FakeOllama()), rows, ck, concurrency=2)
    final = ids_in(ck)
    assert sorted(final) == sorted(r["custom_id"] for r in rows)
    assert len(final) == len(set(final)), "a row was answered twice"


def test_unpulled_model_stops_the_round_immediately(tmp_path, rows):
    from provider.local import LocalServerError, run_rows

    ck = tmp_path / "results.jsonl"
    with pytest.raises(LocalServerError, match="ollama pull"):
        run_rows(client(FakeOllama(pulled=set())), rows, ck)
    assert not ck.exists() or ck.read_text() == ""


def test_rejected_request_is_recorded_as_a_parse_failure(tmp_path, rows):
    """A 400 is deterministic - it is the outcome for that row, kept under C6."""
    from datetime import UTC, datetime

    from provider.collect import collect_results
    from provider.local import run_rows

    target = rows[0]
    target["body"]["messages"][-1]["content"] = "reject:x"
    ck = tmp_path / "results.jsonl"
    run_rows(client(FakeOllama(reject_ids={"x"})), rows, ck)
    got, report = collect_results(
        ck, {r["custom_id"] for r in rows}, "m", "d", "backend", datetime.now(UTC)
    )
    report.raise_if_dirty()
    bad = [c for c in got if not c.parse_ok]
    assert len(bad) == 1 and f"{bad[0].query_id}|{bad[0].sample_idx}" in target["custom_id"]


def test_torn_last_line_from_a_crash_is_dropped_and_retried(tmp_path, rows):
    from provider.local import run_rows

    ck = tmp_path / "results.jsonl"
    run_rows(client(FakeOllama()), rows[:3], ck)
    with ck.open("a") as fh:
        fh.write('{"custom_id": "eli5_00003|0|hea')  # crash mid-write
    run_rows(client(FakeOllama()), rows, ck)
    assert sorted(ids_in(ck)) == sorted(r["custom_id"] for r in rows)


def test_checkpoint_from_another_plan_is_refused(tmp_path, rows):
    from provider.local import run_rows

    ck = tmp_path / "results.jsonl"
    ck.write_text(json.dumps({"custom_id": "other_00000|0|healthy|r999"}) + "\n")
    with pytest.raises(ValueError, match="not in this round"):
        run_rows(client(FakeOllama()), rows, ck)


def test_generate_round_stores_once_with_a_real_digest(tmp_path, rows):
    from provider.collect import read_completions
    from provider.local import generate_round

    fake, out = FakeOllama(), tmp_path / "completions.parquet"
    for _ in range(2):  # the second run must recognise the round is stored
        generate_round(
            client(fake),
            rows,
            tmp_path / "r001",
            out,
            round_id="r001",
            condition="healthy",
            model_id="m",
            digest="a" * 64,
            digest_source="backend",
        )
    got = read_completions(out)
    assert len(got) == len(rows)
    assert all(c.parse_ok and c.digest_source == "backend" for c in got)
    record = json.loads((tmp_path / "r001" / "round.json").read_text())
    assert record["ollama_version"] == "0.0.0-fake" and record["n_rows"] == len(rows)


def test_catalogue_resolves_roles_against_the_local_server_with_real_digests():
    from provider.catalogue import resolve_roles

    for resolved in resolve_roles(client(FakeOllama())).values():
        assert resolved.digest_source == "backend"
        assert len(resolved.digest) == 64


# ── the CLI on the local backend ───────────────────────────────────────────


@pytest.fixture
def local_cli(tmp_path, monkeypatch):
    """Data dir in tmp, a 6-query corpus, and the CLI wired to a fake Ollama."""
    import cli
    from corpus.build import write_queries
    from corpus.schemas import Query
    from provider.local import LocalClient
    from settings import get_settings

    monkeypatch.setenv("DRIFTWATCH_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    queries = [
        Query(query_id=f"eli5_{i:05d}", slice="eli5", prompt=f"why {i}?", split="calib")
        for i in range(3)
    ] + [
        Query(
            query_id=f"gsm8k_{i:05d}",
            slice="gsm8k",
            prompt=f"{i}+1?",
            ground_truth=str(i + 1),
            split="calib",
        )
        for i in range(3)
    ]
    write_queries(queries, get_settings().queries_path)
    fake = FakeOllama()
    monkeypatch.setattr(
        cli,
        "_local_client",
        lambda timeout=None: LocalClient("http://localhost:11434/v1", transport=fake.transport),
    )
    return fake, tmp_path


def run(*args: str) -> object:
    from typer.testing import CliRunner

    from cli import app

    return CliRunner().invoke(app, ["--plain", *args])


def test_generate_then_embed_end_to_end(local_cli, settings):
    from provider.collect import read_completions, read_embeddings
    from settings import get_settings

    _fake, data = local_cli
    result = run("generate", "--condition", "healthy", "--round-id", "r001")
    assert result.exit_code == 0, result.output
    assert run("generate", "--condition", "healthy", "--round-id", "r001").exit_code == 0

    rows = read_completions(get_settings().completions_path)
    assert len(rows) == 6 * get_settings().k, "a rerun must not duplicate the round"
    assert {r.digest_source for r in rows} == {"backend"}

    result = run("embed", "--condition", "healthy", "--round-id", "e001")
    assert result.exit_code == 0, result.output
    assert len(read_embeddings(data / "embeddings.parquet")) == len(rows)


def test_generate_refuses_a_build_that_differs_from_the_pin(local_cli):
    from settings import model_id_for_role

    fake, _ = local_cli
    fake.digest_for[model_id_for_role("healthy")] = "b" * 64
    result = run("generate", "--condition", "healthy", "--round-id", "r001")
    assert result.exit_code != 0
    assert "pins" in str(result.exception)


def test_local_probe_passes_when_seeds_are_honoured(local_cli):
    result = run("probe")
    assert result.exit_code == 0, result.output
    assert "PROBE PASSED" in result.output


def test_local_probe_fails_when_seeds_are_ignored(local_cli):
    fake, _ = local_cli
    fake.answer = lambda body: "always the same"
    assert run("probe").exit_code == 1


def test_local_probe_is_inconclusive_when_the_model_is_missing(local_cli):
    fake, _ = local_cli
    fake.pulled = set()
    result = run("probe")
    assert result.exit_code != 0

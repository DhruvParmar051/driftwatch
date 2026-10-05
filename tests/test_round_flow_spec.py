"""SPEC - a complete round end to end against an in-memory backend.

The unit tests in test_provider_spec / test_manifest_spec pin individual
failure modes. These run the whole sequence - expand, balance into chunks,
manifest, submit, poll, collect, reconcile - because `resume` is the code that
runs unattended for a day, and its bugs only appear when the pieces meet.
"""

from __future__ import annotations

import pytest

from tests.fake_backend import FakeBatchBackend

pytestmark = pytest.mark.spec


@pytest.fixture
def queries():
    from corpus.schemas import Query

    out = []
    for i in range(6):
        out.append(
            Query(
                query_id=f"gsm8k_{i:05d}",
                slice="gsm8k",
                prompt=f"q{i}",
                ground_truth="1",
                split="fault",
            )
        )
        out.append(Query(query_id=f"eli5_{i:05d}", slice="eli5", prompt=f"e{i}", split="fault"))
    return out


def _plan(tmp_path, queries, settings, n_chunks=3) -> tuple:
    from provider import manifest
    from provider.expand import expand_queries
    from provider.submit import balance_chunks

    rows = expand_queries(
        queries,
        settings.k,
        "m",
        "F1",
        "r001",
        None,
        settings.seed_base,
        settings.temperature,
        settings.max_tokens,
    )
    chunks = balance_chunks(rows, n_chunks)
    path = tmp_path / "r001" / "manifest.json"
    manifest.create(path, "r001", "F1", "m", "d" * 64, chunks)
    return path, rows


def _client(backend) -> object:
    from provider.client import BatchClient

    return BatchClient("http://test.invalid/v1", "k", transport=backend.transport)


@pytest.fixture(autouse=True)
def _fast_polling(monkeypatch) -> None:
    monkeypatch.setenv("DRIFTWATCH_POLL_INTERVAL_S", "0")
    from settings import get_settings

    get_settings.cache_clear()


def test_full_round_collects_every_row_once(tmp_path, queries, settings):
    from provider import manifest
    from provider.collect import read_completions

    path, rows = _plan(tmp_path, queries, settings)
    backend = FakeBatchBackend()
    out = tmp_path / "completions.parquet"

    m = manifest.resume(path, _client(backend), out)

    assert m.is_complete
    got = read_completions(out)
    assert len(got) == len(rows)
    assert {(c.query_id, c.sample_idx) for c in got} == {
        (q.query_id, i) for q in queries for i in range(settings.k)
    }
    sent = {r["custom_id"]: r["body"]["seed"] for r in rows}
    assert all(c.seed == sent[f"{c.query_id}|{c.sample_idx}|F1|r001"] for c in got)
    assert all(c.digest == "d" * 64 and c.digest_source == "backend" for c in got)


def test_balanced_chunks_never_split_a_query(tmp_path, queries, settings):
    from collections import defaultdict

    from provider import manifest

    path, _ = _plan(tmp_path, queries, settings)
    homes = defaultdict(set)
    for chunk in manifest.read(path).chunks:
        for cid in chunk.custom_ids:
            homes[cid.split("|")[0]].add(chunk.chunk_idx)
    assert all(len(h) == 1 for h in homes.values())


def test_resume_twice_does_not_duplicate_rows(tmp_path, queries, settings):
    from provider import manifest
    from provider.collect import read_completions

    path, rows = _plan(tmp_path, queries, settings)
    backend = FakeBatchBackend()
    out = tmp_path / "completions.parquet"
    manifest.resume(path, _client(backend), out)
    manifest.resume(path, _client(backend), out)

    assert len(read_completions(out)) == len(rows)
    assert backend.created == len(manifest.read(path).chunks), "a chunk was resubmitted"


def test_crash_after_create_before_record_is_adopted_not_resubmitted(tmp_path, queries, settings):
    """The expensive bug, end to end: the batch exists remotely, the manifest never heard."""
    from provider import manifest
    from provider.submit import submit_batch

    path, _ = _plan(tmp_path, queries, settings)
    backend = FakeBatchBackend()
    client = _client(backend)
    # Simulate the crash: chunk 0 was submitted but its id never reached the manifest.
    submit_batch(client, manifest.chunk_file(path, 0), "/v1/chat/completions", "r001", "F1")
    assert backend.created == 1

    manifest.resume(path, client, tmp_path / "completions.parquet")
    assert backend.created == len(manifest.read(path).chunks), "orphaned chunk was resubmitted"


def test_crash_after_append_before_mark_does_not_double_count(tmp_path, queries, settings):
    from provider import manifest
    from provider.collect import read_completions

    path, rows = _plan(tmp_path, queries, settings)
    backend = FakeBatchBackend()
    out = tmp_path / "completions.parquet"
    manifest.resume(path, _client(backend), out)

    # Rewind chunk 0 to `submitted` as if the process died after the parquet append.
    m = manifest.read(path)
    m.chunks[0].status = "submitted"
    manifest._write(path, m)
    manifest.resume(path, _client(backend), out)

    assert len(read_completions(out)) == len(rows)


def test_worker_error_rows_are_kept_as_parse_failures(tmp_path, queries, settings):
    """C6: an error lands in the ERROR file; it must still come back as a row."""
    from provider import manifest
    from provider.collect import read_completions

    path, rows = _plan(tmp_path, queries, settings)
    failed = rows[0]["custom_id"]
    backend = FakeBatchBackend(error_ids={failed})
    out = tmp_path / "completions.parquet"
    manifest.resume(path, _client(backend), out)

    got = {f"{c.query_id}|{c.sample_idx}|{c.condition}|r001": c for c in read_completions(out)}
    assert len(got) == len(rows)
    assert got[failed].parse_ok is False and got[failed].text == ""


def test_silently_skipped_row_fails_the_chunk_loudly(tmp_path, queries, settings):
    """C10: the daemon skips malformed lines; a short chunk must never be stored."""
    from provider import manifest
    from provider.collect import ReconciliationError

    path, rows = _plan(tmp_path, queries, settings)
    backend = FakeBatchBackend(drop_ids={rows[5]["custom_id"]})
    out = tmp_path / "completions.parquet"
    with pytest.raises(ReconciliationError):
        manifest.resume(path, _client(backend), out)
    assert any(c.status == "failed" for c in manifest.read(path).chunks)


def test_embedding_round_writes_a_vector_per_row(tmp_path, settings):
    from provider import manifest
    from provider.collect import read_embeddings
    from provider.expand import build_embedding_row

    ids = [f"eli5_{i:05d}|{j}|F1|r001" for i in range(3) for j in range(settings.k)]
    rows = [build_embedding_row(f"text {cid}", cid, "emb") for cid in ids]
    path = tmp_path / "e001" / "manifest.json"
    manifest.create(path, "e001", "F1", "emb", "d" * 64, [rows])
    out = tmp_path / "embeddings.parquet"

    manifest.resume(path, _client(FakeBatchBackend()), out)
    vectors = read_embeddings(out)
    assert set(vectors) == set(ids)

"""SPEC - track T2, round manifest and resume.

A round spans up to 24 hours and several chunks. The manifest is what makes an
interrupted round recoverable instead of a wasted day, so these tests pin the
failure modes rather than the happy path.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.spec


@pytest.fixture
def rows():
    return [{"custom_id": f"q{i}|0|healthy|r1", "method": "POST",
             "url": "/v1/chat/completions", "body": {"model": "m", "messages": []}}
            for i in range(10)]


def test_idempotency_key_depends_only_on_rows(rows):
    """It must survive a restart, so nothing process-local may leak in."""
    from provider.manifest import chunk_idempotency_key

    assert chunk_idempotency_key(rows) == chunk_idempotency_key(list(rows))


def test_idempotency_key_changes_with_content(rows):
    from provider.manifest import chunk_idempotency_key

    other = [dict(r) for r in rows]
    other[0]["custom_id"] = "different|0|healthy|r1"
    assert chunk_idempotency_key(other) != chunk_idempotency_key(rows)


def test_key_is_stable_across_processes(rows):
    """Python's salted hash() would pass an in-process test and fail in reality."""
    import json
    import subprocess
    import sys

    code = (
        "import json,sys;"
        "from provider.manifest import chunk_idempotency_key;"
        "print(chunk_idempotency_key(json.load(sys.stdin)))"
    )
    outs = []
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", code], input=json.dumps(rows),
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
        outs.append(proc.stdout.strip())
    assert outs[0] == outs[1]


def test_manifest_is_written_before_anything_is_submitted(tmp_path, rows):
    """A manifest written after submission cannot describe a crash during it."""
    from provider import manifest

    path = tmp_path / "round.json"
    m = manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows[:5], rows[5:]])
    assert path.exists()
    assert len(m.chunks) == 2
    assert all(c.status == "pending" for c in m.chunks)
    assert all(c.idempotency_key for c in m.chunks)


def test_resume_only_sends_pending_chunks(tmp_path, rows):
    from provider import manifest
    from provider.submit import SubmittedBatch

    path = tmp_path / "round.json"
    manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows[:5], rows[5:]])
    manifest.record_submitted(path, 0, SubmittedBatch(
        batch_id="b0", input_file_id="f0", endpoint="/v1/chat/completions", n_rows=5,
        round_id="r001", condition="healthy", submitted_at="2026-09-01T00:00:00Z",
        row_digest="sha256:x"))

    m = manifest.read(path)
    assert [c.chunk_idx for c in m.pending_chunks()] == [1]
    assert [c.chunk_idx for c in m.awaiting_chunks()] == [0]


def test_collected_chunk_is_not_resubmitted(tmp_path, rows):
    """Re-collecting an appended chunk would double its rows in every window."""
    from provider import manifest
    from provider.submit import SubmittedBatch

    path = tmp_path / "round.json"
    manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows])
    manifest.record_submitted(path, 0, SubmittedBatch(
        batch_id="b0", input_file_id="f0", endpoint="/v1/chat/completions", n_rows=10,
        round_id="r001", condition="healthy", submitted_at="2026-09-01T00:00:00Z",
        row_digest="sha256:x"))
    manifest.record_collected(path, 0, n_collected=10)

    m = manifest.read(path)
    assert m.pending_chunks() == []
    assert m.awaiting_chunks() == []
    assert m.is_complete


def test_round_is_incomplete_while_any_chunk_is_outstanding(tmp_path, rows):
    from provider import manifest

    path = tmp_path / "round.json"
    m = manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows[:5], rows[5:]])
    assert not m.is_complete


def test_orphaned_chunk_is_adopted_not_resubmitted(tmp_path, rows):
    """THE EXPENSIVE BUG.

    A crash between creating a batch and recording its id leaves a chunk marked
    `submitting` while the job may be running right now. Resubmitting wastes a
    day of GPU and returns duplicate rows.
    """
    import httpx

    from provider import manifest
    from provider.client import BatchClient

    path = tmp_path / "round.json"
    m = manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows])
    m.chunks[0].status = "submitting"
    key = m.chunks[0].idempotency_key

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={
        "data": [{"id": "b-orphan", "status": "in_progress",
                  "metadata": {"round_id": "r001", "idempotency_key": key}}]
    }))
    client = BatchClient("http://test.invalid/v1", "k", transport=transport)

    recovered = manifest.recover_orphans(path, client)
    assert recovered.chunks[0].status == "submitted"
    assert recovered.chunks[0].batch_id == "b-orphan"


def test_unmatched_orphan_is_reset_to_pending(tmp_path, rows):
    import httpx

    from provider import manifest
    from provider.client import BatchClient

    path = tmp_path / "round.json"
    m = manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows])
    m.chunks[0].status = "submitting"

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"data": []}))
    client = BatchClient("http://test.invalid/v1", "k", transport=transport)

    assert manifest.recover_orphans(path, client).chunks[0].status == "pending"


def test_manifest_write_is_atomic(tmp_path, rows):
    """A half-written manifest looks valid and misreports what was sent."""
    from provider import manifest

    path = tmp_path / "round.json"
    manifest.create(path, "r001", "healthy", "model-x", "d" * 64, [rows])
    for _ in range(3):
        manifest.record_collected(path, 0, n_collected=10)
        manifest.read(path)  # must always parse


def test_read_missing_manifest_raises(tmp_path):
    from provider import manifest

    with pytest.raises((FileNotFoundError, OSError)):
        manifest.read(tmp_path / "nope.json")

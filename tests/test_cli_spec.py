"""SPEC - the CLI, end to end, offline.

Every command runs through typer against the in-memory batch backend, with the
data directory redirected to a temp dir. These are the commands a teammate
actually types; each one was a stub until it was wired, so each gets run.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from tests.fake_backend import FakeBatchBackend

pytestmark = pytest.mark.spec


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DRIFTWATCH_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DRIFTWATCH_POLL_INTERVAL_S", "0")
    from settings import get_settings

    get_settings.cache_clear()
    return tmp_path


@pytest.fixture
def backend(monkeypatch):
    import cli
    from provider.client import BatchClient

    fake = FakeBatchBackend()
    monkeypatch.setattr(
        cli, "_client", lambda: BatchClient("http://test.invalid/v1", "k", transport=fake.transport)
    )
    return fake


@pytest.fixture
def full_corpus(data_dir):
    from corpus.build import build_corpus, write_queries
    from corpus.splits import assign_splits
    from settings import get_settings

    s = get_settings()
    queries = assign_splits(build_corpus(s.corpus_seed), s.corpus_seed)
    write_queries(queries, s.queries_path)
    return queries


@pytest.fixture
def small_corpus(data_dir):
    from corpus.build import write_queries
    from corpus.schemas import Query
    from settings import get_settings

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
    return queries


def run(*args: str) -> object:
    from cli import app

    return CliRunner().invoke(app, ["--plain", *args])


def test_probe_passes_when_the_backend_honours_seeds(small_corpus, backend):
    result = run("probe", "--rows", "20", "--backend", "batch")
    assert result.exit_code == 0, result.output
    assert "PROBE PASSED" in result.output


def test_probe_fails_loudly_when_seeds_are_ignored(small_corpus, backend):
    backend.answer = lambda body: "the same text whatever the seed"
    result = run("probe", "--rows", "20", "--backend", "batch")
    assert result.exit_code == 1
    assert "PROBE FAILED" in result.output


def test_expand_submit_resume_produces_a_reconciled_round(small_corpus, backend, data_dir):
    from provider.collect import read_completions
    from settings import get_settings

    jsonl = data_dir / "r001.jsonl"
    assert (
        run("expand", "--condition", "healthy", "--round-id", "r001", "--out", str(jsonl)).exit_code
        == 0
    )
    result = run(
        "submit", str(jsonl), "--condition", "healthy", "--round-id", "r001", "--chunks", "2"
    )
    assert result.exit_code == 0, result.output
    result = run("resume", "r001")
    assert result.exit_code == 0, result.output

    rows = read_completions(get_settings().completions_path)
    assert len(rows) == len(small_corpus) * get_settings().k
    assert {r.digest_source for r in rows} == {"backend"}


def test_expand_refuses_f4(small_corpus, data_dir):
    result = run("expand", "--condition", "F4", "--round-id", "r9", "--out", str(data_dir / "x"))
    assert result.exit_code != 0


def test_calibrate_can_leave_out_constant_signals(full_corpus, data_dir):
    """A healthy model that never refuses (S2) or never breaks JSON (S5) leaves that
    signal constant at zero: nothing to freeze, so it must be possible to leave it out.
    """
    from provider.collect import write_completions
    from settings import get_settings
    from synthetic.generate import DegradationKnob, generate_completions

    s = get_settings()
    rows = generate_completions(full_corpus, DegradationKnob(), s.k, "healthy", s.seed_base)
    write_completions(rows, s.completions_path, mode="overwrite")

    comp = ("--completions", str(s.completions_path), "--no-embeddings")
    drop = ("--without", "refusal_rate", "--without", "format_validity")
    result = run("calibrate", *comp, *drop)
    assert result.exit_code == 0, result.output
    proposal = json.loads((data_dir / "thresholds.proposed.json").read_text())
    assert proposal["signals"] == ["length_ks", "self_consistency", "lexical_ttr", "label_entropy"]
    assert proposal["signals"] == list(proposal["thresholds"])

    assert run("calibrate", *comp, "--without", "no_such_signal").exit_code != 0


def test_per_slice_v2_freezes_to_its_own_file_and_says_so(full_corpus, data_dir, monkeypatch):
    """v2 is post-hoc: it must never land in, or be confused with, v1's record."""
    from detect import freeze
    from provider.collect import write_completions
    from settings import get_settings
    from synthetic.generate import DegradationKnob, generate_completions

    monkeypatch.setattr(freeze, "current_git_commit", lambda: "test0000commit")
    s = get_settings()
    rows = generate_completions(full_corpus, DegradationKnob(), s.k, "healthy", s.seed_base)
    write_completions(rows, s.completions_path, mode="overwrite")
    comp = str(s.completions_path)
    proposal, frozen = data_dir / "v2.proposed.json", data_dir / "thresholds_v2.json"

    drop = ("--no-embeddings", "--without", "refusal_rate", "--without", "format_validity")
    result = run("calibrate", "--completions", comp, *drop, "--per-slice", "--out", str(proposal))
    assert result.exit_code == 0, result.output
    assert json.loads(proposal.read_text())["scoring"] == "per_slice"

    args = ("--thresholds", str(proposal), "--completions", comp, "--out", str(frozen))
    assert run("freeze", *args).exit_code == 0
    assert json.loads(frozen.read_text())["scoring"] == "per_slice"
    assert not s.thresholds_path.exists(), "v2 must not take v1's place"


def test_calibrate_freeze_verify_evaluate_report(full_corpus, data_dir, monkeypatch):
    """The whole Stage 5-8 path on synthetic rounds, through the CLI."""
    from corpus.splits import fault_subset
    from detect import freeze
    from provider.collect import write_completions
    from settings import get_settings
    from synthetic.generate import DegradationKnob, generate_completions

    monkeypatch.setattr(freeze, "current_git_commit", lambda: "test0000commit")
    s = get_settings()
    fault = fault_subset(full_corpus)
    rows = generate_completions(full_corpus, DegradationKnob(), s.k, "healthy", s.seed_base)
    rows += generate_completions(fault, DegradationKnob.uniform(0.6), s.k, "F1", s.seed_base)
    write_completions(rows, s.completions_path, mode="overwrite")
    comp = str(s.completions_path)

    result = run("calibrate", "--completions", comp, "--no-embeddings")
    assert result.exit_code == 0, result.output
    proposal = data_dir / "thresholds.proposed.json"
    assert set(json.loads(proposal.read_text())["thresholds"]) == {
        "length_ks",
        "refusal_rate",
        "self_consistency",
        "lexical_ttr",
        "format_validity",
        "label_entropy",
    }

    assert run("freeze", "--thresholds", str(proposal), "--completions", comp).exit_code == 0
    assert run("freeze", "--thresholds", str(proposal), "--completions", comp).exit_code != 0, (
        "a second freeze must refuse to overwrite the first"
    )
    assert run("verify", "--completions", comp).exit_code == 0

    result = run("evaluate", "--completions", comp, "--n-runs", "4")
    assert result.exit_code == 0, result.output
    assert "Stage 5 gate: PASSED" in result.output
    results = data_dir / "results" / "evaluation.json"
    data = json.loads(results.read_text())
    assert {r["condition"] for r in data["summary"]} == {"F1", "F4"}

    out = data_dir / "report"
    assert run("report", "--results", str(results), "--out-dir", str(out)).exit_code == 0
    assert (out / "report.md").exists() and (out / "summary.csv").exists()


def test_calibrate_refuses_any_split_but_calib(full_corpus, data_dir):
    from provider.collect import write_completions
    from settings import get_settings
    from synthetic.generate import DegradationKnob, generate_completions

    s = get_settings()
    write_completions(
        generate_completions(full_corpus[:50], DegradationKnob(), s.k, "healthy", 1),
        s.completions_path,
        mode="overwrite",
    )
    result = run(
        "calibrate",
        "--completions",
        str(s.completions_path),
        "--split",
        "heldout",
        "--no-embeddings",
    )
    assert result.exit_code != 0


def test_evaluate_refuses_without_a_frozen_file(full_corpus, data_dir):
    from provider.collect import write_completions
    from settings import get_settings
    from synthetic.generate import DegradationKnob, generate_completions

    s = get_settings()
    write_completions(
        generate_completions(full_corpus[:50], DegradationKnob(), s.k, "healthy", 1),
        s.completions_path,
        mode="overwrite",
    )
    assert run("evaluate", "--completions", str(s.completions_path)).exit_code != 0


def test_synthetic_evaluation_runs_without_a_freeze(full_corpus, data_dir):
    """Stage 7 gate path: synthetic rounds, calibrated in memory, never frozen."""
    result = run("evaluate", "--completions", "unused.parquet", "--synthetic", "--n-runs", "3")
    assert result.exit_code == 0, result.output
    data = json.loads((data_dir / "results" / "evaluation_synthetic.json").read_text())
    assert data["mode"] == "synthetic"
    assert {r["condition"] for r in data["summary"]} == {"F1", "F2", "F3", "F4"}
    assert not (data_dir / "thresholds.json").exists(), "synthetic mode must not freeze"


def test_probe_is_inconclusive_when_requests_fail(small_corpus, backend):
    """Failed requests come back empty - and empty answers are all 'identical'.
    Judging seeds on them would blame the seed for a backend failure.
    """
    backend.error_ids = {f"eli5_00000|{i}|probe|probe" for i in range(20)}
    result = run("probe", "--rows", "20", "--backend", "batch")
    assert result.exit_code == 2, result.output
    assert "PROBE INCONCLUSIVE" in result.output
    assert "seeds were NOT tested" in result.output

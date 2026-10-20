"""Command-line interface. Every command wired to the library.

Commands run in the Stage order of the build plan. Each one is thin: parse
arguments, configure logging, call into the library, report. Put no logic here
that a test would want to reach directly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from log import configure_logging, get_logger
from version import __version__

if TYPE_CHECKING:
    from collections.abc import Sequence

    from provider.catalogue import ResolvedModel
    from provider.client import BatchClient
    from provider.local import LocalClient

app = typer.Typer(
    name="driftwatch",
    help="Label-free detection of silent quality degradation in deployed language models.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    plain: bool = typer.Option(False, "--plain", help="Human-readable logs instead of JSON."),
) -> None:
    """Configure logging for every subcommand."""
    configure_logging(level="DEBUG" if verbose else "INFO", json_output=not plain)


@app.command()
def version() -> None:
    """Print the driftwatch version."""
    typer.echo(__version__)


# ── Stage 0-1: corpus and synthetic data ───────────────────────────────────

corpus_app = typer.Typer(help="Build the query corpus and its splits (Stage 1).")
app.add_typer(corpus_app, name="corpus")


@corpus_app.command("build")
def corpus_build(
    out: Path = typer.Option(None, help="Output queries.jsonl. Defaults to settings.queries_path."),
    seed: int | None = typer.Option(None, help="Corpus seed. Defaults to settings.corpus_seed."),
) -> None:
    """Assemble the 5,000-query corpus and assign splits.

    Gate: exactly 5,000 rows; 2,500 calib / 1,000 heldout / 1,500 fault; slice
    counts match settings.SPLIT_SIZES; splits disjoint; 2,800 rows have
    ground_truth; the draw is reproducible from the recorded seed.
    """
    from corpus.build import build_corpus, write_queries
    from corpus.splits import assign_splits
    from settings import get_settings

    s = get_settings()
    seed = seed if seed is not None else s.corpus_seed
    out = out or s.queries_path
    queries = assign_splits(build_corpus(seed), seed)
    n = write_queries(queries, out)
    log.info("corpus.built", rows=n, path=str(out), seed=seed)


@app.command()
def synth(
    knob: list[float] = typer.Option([0.0], help="Degradation level(s), 0.0-1.0. Repeatable."),
    k: int | None = typer.Option(None, help="Samples per query."),
    condition: str = typer.Option("healthy", help="Condition label for the generated rows."),
    out_dir: Path = typer.Option(None, help="Output directory."),
) -> None:
    """Generate synthetic completions (Stage 0).

    The insurance policy: every later stage is developed against this output, so
    a bug never costs a submitted round.
    """
    from corpus.build import read_queries
    from settings import get_settings
    from synthetic.generate import DegradationKnob, write_synthetic_round

    s = get_settings()
    out_dir = out_dir or s.data_dir
    queries = read_queries(s.queries_path)
    for value in knob:
        path = out_dir / f"synthetic_knob{value:.2f}_{condition}.parquet"
        n = write_synthetic_round(
            queries, DegradationKnob.uniform(value), k or s.k, condition, s.seed_base, path
        )
        log.info("synth.written", knob=value, rows=n, path=str(path))


# ── Stage 2-3: provider ────────────────────────────────────────────────────


@app.command()
def probe(
    rows: int = typer.Option(20, help="Total probe rows (10 distinct + 10 identical seeds)."),
    backend: str = typer.Option("local", help="'local' (Ollama) or 'batch' (Sheshnag)."),
    concurrency: int = typer.Option(4, help="Local only: requests in flight at once."),
) -> None:
    """Stage 2 seed-determinism probe. RUN THIS BEFORE ANY LARGE ROUND.

    Sends one prompt x 10 distinct seeds and the same prompt x 10 identical
    seeds to the HEALTHY model, then asserts: distinct seeds give >= 8 distinct
    outputs, identical seeds give identical output.

    Locally, run it at the concurrency you will generate with: identical seeds
    can stop reproducing when Ollama batches parallel requests together, and
    this is where that shows up. A cold prompt cache also changes the text
    for a fixed seed, so the local probe warms the cache with one unscored
    request before the scored rows.

    IF DISTINCT SEEDS DO NOT VARY THE OUTPUT, STOP. S3 (self-consistency) is not
    viable and the design must change before a baseline round is spent.

    Exit codes: 0 passed; 1 seeds not honoured; 2 inconclusive - requests
    failed on the backend, so seeds were never tested (the backend's error is
    printed).
    """
    import tempfile

    from corpus.build import read_queries
    from provider.collect import collect_results
    from provider.expand import build_chat_row
    from settings import get_settings, system_prompt_for_condition

    s = get_settings()
    if rows < 4 or rows % 2:
        raise typer.BadParameter("rows must be an even number >= 4")
    half = rows // 2
    resolved = _resolve_role("healthy", backend)

    # One free-text query from the corpus: long answers make sampling variation visible.
    query = next(q for q in read_queries(s.queries_path) if q.slice == "eli5")
    system = system_prompt_for_condition("healthy")
    probe_rows = []
    for i in range(rows):
        row = build_chat_row(
            query,
            i,
            resolved.model_id,
            "probe",
            "probe",
            system,
            s.seed_base,
            s.temperature,
            s.max_tokens,
        )
        if i > half:  # second half: every row reuses the seed of row `half`
            row["body"]["seed"] = probe_rows[half]["body"]["seed"]
        probe_rows.append(row)

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "probe.out.jsonl"
        _answer_rows(probe_rows, out, backend, concurrency)
        seeds = {r["custom_id"]: r["body"]["seed"] for r in probe_rows}
        completions, report = collect_results(
            out,
            set(seeds),
            resolved.model_id,
            resolved.digest,
            resolved.digest_source,
            _now(),
            seeds=seeds,
        )
    report.raise_if_dirty()
    failed = [c for c in completions if not c.parse_ok]
    if failed:
        # Empty answers are all "identical", so judging seeds on them would
        # report a seed problem that is really a backend problem.
        typer.echo(
            f"PROBE INCONCLUSIVE: {len(failed)}/{rows} requests failed on the backend, so "
            "seeds were NOT tested. The backend said:"
        )
        for sample in report.error_samples or ["(no error text - see the log)"]:
            typer.echo(f"  {sample}")
        raise typer.Exit(code=2)
    texts = {c.sample_idx: c.text for c in completions}
    distinct = len({texts[i] for i in range(half)})
    identical = len({texts[i] for i in range(half, rows)})
    ok = distinct >= int(0.8 * half) and identical == 1
    log.info(
        "probe.result",
        backend=backend,
        model=resolved.model_id,
        distinct_seed_outputs=distinct,
        of=half,
        identical_seed_outputs=identical,
        passed=ok,
    )
    typer.echo(f"distinct seeds -> {distinct}/{half} distinct outputs (need >= {int(0.8 * half)})")
    typer.echo(f"identical seed -> {identical} distinct output(s) (need exactly 1)")
    if not ok:
        typer.echo("PROBE FAILED: `seed` is not honoured as required. S3 is not viable - STOP.")
        raise typer.Exit(code=1)
    typer.echo("PROBE PASSED: seeds are honoured. The baseline round may be run.")


@app.command()
def expand(
    condition: str = typer.Option(..., help="healthy, F1..F5."),
    round_id: str = typer.Option(..., help="Round identifier, e.g. r003."),
    out: Path = typer.Option(..., help="Output JSONL path."),
) -> None:
    """Expand queries into k JSONL rows each, with distinct seeds (Stage 3).

    healthy runs the whole corpus; F1, F2, F3 and F5 run the ONE shared fault
    subset (C8). F4 is not generated: it is a re-weighted view over healthy
    completions, derived at evaluation time.
    """
    from provider.expand import write_jsonl

    rows, _role = _round_rows(condition, round_id)
    n = write_jsonl(rows, out)
    log.info("expand.written", condition=condition, round_id=round_id, rows=n, path=str(out))


@app.command()
def generate(
    condition: str = typer.Option(..., help="healthy, F1, F2, F3, F5, F7, F8 or H2-H5."),
    round_id: str = typer.Option(..., help="Round identifier, e.g. r001."),
    concurrency: int = typer.Option(4, help="Requests in flight; match OLLAMA_NUM_PARALLEL."),
    timeout: float = typer.Option(900.0, help="Seconds allowed per request."),
) -> None:
    """Generate a round on the LOCAL Ollama server, checkpointing every answer (Stage 3).

    Rerun the exact same command after any interruption: answers already on
    disk are kept and only the rest are sent. When every row is answered the
    round is reconciled (C10) and appended to completions.parquet once.

    Refuses to run if the model the server reports is not the build pinned in
    config/models.yaml (C9).
    """
    from provider.local import generate_round
    from settings import get_settings

    s = get_settings()
    rows, role = _round_rows(condition, round_id)
    resolved = _resolve_role(role, "local")
    client = _local_client(timeout)
    generate_round(
        client,
        rows,
        _rounds_dir() / round_id,
        s.completions_path,
        round_id=round_id,
        condition=condition,
        model_id=resolved.model_id,
        digest=resolved.digest,
        digest_source=resolved.digest_source,
        concurrency=concurrency,
    )
    typer.echo(
        f"round {round_id} ({condition}, {resolved.model_id}) complete -> {s.completions_path}"
    )


@app.command()
def submit(
    jsonl: Path = typer.Argument(..., help="JSONL file to submit."),
    condition: str = typer.Option(...),
    round_id: str = typer.Option(...),
    chunks: int = typer.Option(4, help="Jobs to split the round into (token-balanced)."),
) -> None:
    """Upload and submit a batch, then persist the submission record (Stage 3).

    The JSONL is split into `--chunks` jobs of roughly equal expected TOKENS -
    workers run one prompt at a time, so parallelism comes from job count. A
    round manifest is written BEFORE anything is uploaded; follow up with
    `driftwatch resume ROUND_ID`, which is safe to rerun after any crash.
    """
    from provider import manifest
    from provider.submit import balance_chunks, read_jsonl_rows

    rows = read_jsonl_rows(jsonl)
    if not rows:
        raise typer.BadParameter(f"{jsonl} has no rows")
    path = _manifest_path(round_id)
    model_id = rows[0]["body"]["model"]
    resolved = _resolve_model(model_id)
    manifest.create(
        path, round_id, condition, model_id, resolved.digest, balance_chunks(rows, chunks)
    )
    m = manifest.submit_pending(path, _client())
    log.info("submit.round_submitted", round_id=round_id, chunks=len(m.chunks), manifest=str(path))
    typer.echo(f"submitted {len(m.chunks)} chunk(s); manifest {path}")


@app.command()
def collect(batch_id: str = typer.Argument(...), out: Path = typer.Option(...)) -> None:
    """Download, parse and RECONCILE a completed batch (Stage 3).

    Fails loudly on any submitted/returned mismatch (C10). A partially returned
    batch breaks every per-cell sample size and must never be analysed.
    """
    from provider import manifest

    for path in sorted((_rounds_dir()).glob("*/manifest.json")):
        if any(c.batch_id == batch_id for c in manifest.read(path).chunks):
            m = manifest.collect_batch(path, batch_id, _client(), out)
            typer.echo(f"collected {batch_id}; round complete: {m.is_complete}")
            return
    raise typer.BadParameter(f"no round manifest under {_rounds_dir()} contains {batch_id}")


@app.command()
def resume(
    round_id: str = typer.Argument(..., help="Round to continue."),
    out: Path = typer.Option(
        None, help="Output store. Defaults to completions.parquet or embeddings.parquet."
    ),
    retry_failed: bool = typer.Option(
        False, "--retry-failed", help="Return FAILED chunks to pending first. Read their errors."
    ),
) -> None:
    """Continue a round from its manifest: recover orphans, submit, poll, collect (Stage 3).

    Idempotent - rerun it after any crash or interruption. Reconciles each chunk
    on arrival and the whole round at the end (C10).
    """
    from provider import manifest
    from settings import EMBEDDINGS_URL

    path = _manifest_path(round_id)
    if retry_failed:
        manifest.reset_failed(path)
    m = manifest.read(path)
    out = out or (_embeddings_path() if m.endpoint == EMBEDDINGS_URL else _completions_path())
    m = manifest.resume(path, _client(), out)
    typer.echo(f"round {round_id}: complete={m.is_complete} -> {out}")


@app.command()
def embed(
    completions: Path = typer.Option(None, help="Completions parquet. Defaults to settings."),
    condition: str = typer.Option(..., help="Which condition's completions to embed."),
    round_id: str = typer.Option(..., help="Identifier for the embedding round, e.g. e001."),
    backend: str = typer.Option("local", help="'local' (Ollama) or 'batch' (Sheshnag)."),
    concurrency: int = typer.Option(4, help="Local only: requests in flight at once."),
    chunks: int = typer.Option(1, help="Batch only: jobs to split the round into."),
) -> None:
    """Run the S7 embedding round for one condition's completions.

    Parse-failed rows have no text and are not embedded; S7 excludes them
    symmetrically. Locally the round runs to completion here (rerun to resume);
    on the batch backend it is submitted - continue with `driftwatch resume`.
    """
    from provider.collect import read_completions
    from provider.expand import build_embedding_row, encode_custom_id

    resolved = _resolve_role("embedding", backend)
    rows = [
        build_embedding_row(
            c.text,
            encode_custom_id(c.query_id, c.sample_idx, c.condition, round_id),
            resolved.model_id,
        )
        for c in read_completions(completions or _completions_path())
        if c.condition == condition and c.parse_ok
    ]
    if not rows:
        raise typer.BadParameter(f"no parse_ok completions for condition {condition!r}")

    if backend == "local":
        from provider.local import generate_embedding_round

        vectors = generate_embedding_round(
            _local_client(),
            rows,
            _rounds_dir() / round_id,
            _embeddings_path(),
            round_id=round_id,
            condition=condition,
            model_id=resolved.model_id,
            digest=resolved.digest,
            digest_source=resolved.digest_source,
            concurrency=concurrency,
        )
        typer.echo(f"embedded {len(vectors)} rows -> {_embeddings_path()}")
        return

    from provider import manifest
    from provider.submit import balance_chunks

    path = _manifest_path(round_id)
    manifest.create(
        path, round_id, condition, resolved.model_id, resolved.digest, balance_chunks(rows, chunks)
    )
    m = manifest.submit_pending(path, _client())
    typer.echo(f"submitted {len(m.chunks)} embedding chunk(s); manifest {path}")


# ── Stage 5: calibrate and freeze ──────────────────────────────────────────


@app.command()
def calibrate(
    completions: Path = typer.Option(..., help="Healthy completions parquet."),
    split: str = typer.Option("calib", help="Must be 'calib'. Anything else raises LeakageError."),
    out: Path = typer.Option(None, help="Where to write the proposed thresholds."),
    no_embeddings: bool = typer.Option(
        False, "--no-embeddings", help="Calibrate WITHOUT S7 (m=6). Decide before freezing."
    ),
    without: list[str] = typer.Option(
        [],
        "--without",
        help="Leave a signal out (repeatable), e.g. refusal_rate when the healthy model "
        "never refuses: the signal is then constant and has no threshold to freeze. "
        "Changes m; decide before freezing.",
    ),
    scoring: str = typer.Option(
        "pooled",
        help="pooled (v1), per_slice (v2) or per_slice_capped (v3); see signals.stratified. "
        "v2 and v3 are post-hoc: freeze them to their own file with `freeze --out`.",
    ),
) -> None:
    """Fit reference distributions and estimate thresholds (Stage 5).

    Calibrates on split == "calib" ONLY (C4). Thresholds are set at the
    Sidak-corrected per-signal level so the FAMILY-WISE false-alarm rate meets
    the budget - calibrating each signal at the budget itself would run about 7x
    over it at m=7.
    """
    from dataclasses import asdict

    from detect.calibrate import LeakageError, calibrate_with_report, sidak_alpha
    from evaluate.pipeline import build_monitor_signals
    from settings import get_settings

    s = get_settings()
    if split != "calib":
        raise LeakageError(f"thresholds are calibrated on 'calib' only (C4), not {split!r}")
    queries, rows = _calibration_inputs(completions)
    signals = build_monitor_signals(
        _embeddings_or_none(no_embeddings), _enabled(no_embeddings, without), scoring
    )
    thresholds, fits = calibrate_with_report(
        signals, queries, rows, s.false_alarm_budget, s.bootstrap_windows, s.calibration_seed
    )
    proposal = {
        "thresholds": thresholds,
        "per_signal_alpha": sidak_alpha(s.false_alarm_budget, len(signals)),
        "family_wise_budget": s.false_alarm_budget,
        "bootstrap_windows": s.bootstrap_windows,
        "calibration_seed": s.calibration_seed,
        "signals": list(signals),
        "scoring": scoring,
        "tail_fits": {n: asdict(f) for n, f in fits.items()},
    }
    out = out or s.data_dir / "thresholds.proposed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proposal, indent=2) + "\n")
    log.info("calibrate.proposed", path=str(out), signals=list(signals))
    for name, fit in fits.items():
        typer.echo(f"{name:18} {fit.threshold:.6f}  ({fit.method})")


@app.command("freeze")
def freeze_cmd(
    thresholds: Path = typer.Option(..., help="Proposed thresholds from `calibrate`."),
    completions: Path = typer.Option(..., help="The exact calibration rows used."),
    out: Path = typer.Option(
        None, help="Where to freeze. Defaults to settings.thresholds_path (v1's record)."
    ),
) -> None:
    """Freeze thresholds to data/thresholds.json. THE POINT OF NO RETURN (Stage 5).

    Records the git commit and a hash of the calibration rows. Refuses to
    overwrite an existing freeze: to refreeze deliberately, move the old file
    aside by hand so the act is visible in git and shell history.
    """
    from detect import freeze
    from settings import get_settings

    s = get_settings()
    proposal = json.loads(thresholds.read_text())
    _queries, rows = _calibration_inputs(completions)
    path = out or s.thresholds_path
    record = freeze.write(
        path,
        proposal["thresholds"],
        rows,
        proposal["family_wise_budget"],
        proposal["per_signal_alpha"],
        s.k,
        s.window_size,
        s.window_stride,
        proposal["bootstrap_windows"],
        proposal["calibration_seed"],
        proposal.get("scoring", "pooled"),
    )
    typer.echo(f"FROZEN at commit {record.git_commit} ({record.scoring}): {path}")


@app.command()
def verify(thresholds: Path = typer.Option(None), completions: Path = typer.Option(...)) -> None:
    """Verify a frozen thresholds.json against its calibration rows (Stage 5 gate)."""
    from detect import freeze
    from settings import get_settings

    path = thresholds or get_settings().thresholds_path
    _queries, rows = _calibration_inputs(completions)
    record = freeze.verify(path, rows)
    typer.echo(f"OK: {path} matches its {len(rows)} calibration rows (commit {record.git_commit})")


# ── Stage 6-8: oracle, evaluate, report ────────────────────────────────────


@app.command()
def oracle(completions: Path = typer.Option(...), condition: str = typer.Option(...)) -> None:
    """Ground-truth scoring on the verifiable slices (Stage 6).

    Runs strictly AFTER detection. Nothing under detect/ may import oracle/ (C3).
    """
    from corpus.build import read_queries
    from oracle.score import score_condition
    from provider.collect import read_completions
    from settings import get_settings

    queries = read_queries(get_settings().queries_path)
    for r in score_condition(read_completions(completions), queries, condition):
        typer.echo(
            f"{r.condition:8} {r.slice:10} acc={r.accuracy:.4f} "
            f"({r.n_correct}/{r.n_scored}, {r.n_parse_failures} parse failures)"
        )


@app.command()
def evaluate(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(
        None, help="Frozen thresholds. Defaults to settings.thresholds_path."
    ),
    synthetic: bool = typer.Option(
        False, "--synthetic", help="Evaluate against synthetic data (Stage 7 gate)."
    ),
    n_runs: int = typer.Option(200, help="Randomised-onset replay runs per condition."),
    out: Path = typer.Option(None, help="Results file. Defaults to data/results/evaluation.json."),
) -> None:
    """Replay, detection latency, ROC and bootstrap CIs (Stage 7).

    REFUSES TO RUN without a thresholds.json that passes freeze.verify().
    """
    from corpus.build import read_queries
    from detect import freeze
    from detect.calibrate import fit_signals
    from evaluate.pipeline import (
        alias_embeddings,
        build_monitor_signals,
        calibration_rows,
        derive_f4,
        derive_f6,
        evaluate_run,
        write_results,
    )
    from provider.collect import read_completions, read_embeddings
    from settings import get_settings

    s = get_settings()
    if synthetic:
        results = _synthetic_evaluation(n_runs)
        out = s.data_dir / "results" / "evaluation_synthetic.json"
    else:
        path = thresholds or s.thresholds_path
        queries = read_queries(s.queries_path)
        rows = read_completions(completions)
        calib_q, calib_rows = calibration_rows(queries, rows)
        record = freeze.verify(path, calib_rows)  # REFUSES to continue on any mismatch
        names = list(record.thresholds)
        embeddings = None
        if "embedding_shift" in names:
            healthy = [c for c in rows if c.condition == "healthy"]
            embeddings = alias_embeddings(
                read_embeddings(_embeddings_path()),
                derive_f4(queries, healthy, s.corpus_seed),
                "F4",
            )
            embeddings = alias_embeddings(
                embeddings, derive_f6(queries, healthy, s.corpus_seed), "F6"
            )
        signals = fit_signals(
            build_monitor_signals(embeddings, names, record.scoring), calib_q, calib_rows
        )
        results = evaluate_run(
            queries,
            rows,
            signals,
            record.thresholds,
            per_signal_alpha=record.per_signal_alpha,
            n_runs=n_runs,
            seed=s.calibration_seed,
            mode="real",
        )
        results["scoring"] = record.scoring
        out = out or s.data_dir / "results" / "evaluation.json"
    write_results(results, out)
    typer.echo(
        f"Stage 5 gate: {'PASSED' if results['stage5_gate_passed'] else 'FAILED'} "
        f"(family-wise false-alarm rate {results['false_alarm']['family_wise']:.4f})"
    )
    typer.echo(f"results -> {out}")


@app.command()
def report(results: Path = typer.Option(...), out_dir: Path = typer.Option(...)) -> None:
    """Generate results tables and figures (Stage 8).

    Detection latency is plotted as a DISTRIBUTION, never a point estimate, and
    results are reported PER FAULT - never pooled into one success rate.
    """
    from evaluate.pipeline import write_report

    for path in write_report(json.loads(results.read_text()), out_dir):
        typer.echo(f"wrote {path}")


@app.command()
def gradual(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown summary table."),
    n_runs: int = typer.Option(200, help="Replays per fault and schedule."),
) -> None:
    """Partial and gradual degradation against a FROZEN monitor (Stage 7b).

    F1 and F2 served to a share of queries (5% to 100%), or ramped in over 30
    windows. Nothing is calibrated here; the record is verified first, as in
    `evaluate`.
    """
    from corpus.build import read_queries
    from corpus.splits import fault_subset
    from detect import freeze
    from detect.calibrate import fit_signals
    from evaluate.gradual import (
        GRADUAL_FAULTS,
        gradual_markdown,
        run_gradual,
        summarise_gradual,
    )
    from evaluate.pipeline import (
        ORACLE_NOT_A_QUALITY_MEASURE,
        _oracle,
        build_monitor_signals,
        calibration_rows,
    )
    from provider.collect import read_completions, read_embeddings
    from settings import get_settings

    s = get_settings()
    queries = read_queries(s.queries_path)
    rows = read_completions(completions)
    calib_q, calib_rows = calibration_rows(queries, rows)
    record = freeze.verify(thresholds, calib_rows)  # REFUSES to continue on any mismatch
    names = list(record.thresholds)
    embeddings = read_embeddings(_embeddings_path()) if "embedding_shift" in names else None
    signals = fit_signals(
        build_monitor_signals(embeddings, names, record.scoring), calib_q, calib_rows
    )
    healthy = [c for c in rows if c.condition == "healthy"]
    faults = {cond: [c for c in rows if c.condition == cond] for cond in GRADUAL_FAULTS}
    records = run_gradual(
        queries, healthy, faults, signals, record.thresholds, n_runs=n_runs, seed=s.calibration_seed
    )

    oracle = _oracle(queries, rows, list(GRADUAL_FAULTS), fault_subset(queries))

    def mean_accuracy(cond: str) -> float:
        kept = [
            v["accuracy"] for k, v in oracle[cond].items() if k not in ORACLE_NOT_A_QUALITY_MEASURE
        ]
        return sum(kept) / len(kept)

    drop = {c: 100 * (mean_accuracy("healthy") - mean_accuracy(c)) for c in GRADUAL_FAULTS}
    summary = summarise_gradual(records, drop)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"scoring": record.scoring, "summary": summary, "records": records}, indent=2)
        + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(gradual_markdown(summary, record.scoring))
    typer.echo(report_md.read_text())


@app.command()
def sensitivity(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(
        ..., help="A frozen record: its signals and scoring are reused."
    ),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown tables."),
    n_runs: int = typer.Option(200, help="Replay runs per condition."),
    gradual_runs: int = typer.Option(200, help="Gradual replays per fault and schedule."),
) -> None:
    """Calibration size and window size, re-calibrated and replayed (Stage 7e, exploratory).

    Refuses unless the full-data, W=200 run reproduces the frozen thresholds,
    so every other row differs from the frozen monitor in one thing only.
    """
    import math

    from evaluate.pipeline import build_monitor_signals
    from evaluate.sensitivity import (
        calibration_size_study,
        sensitivity_markdown,
        window_size_study,
    )
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ("F4", "F6"))
    names = list(record.thresholds)

    def make_signals() -> dict:
        return build_monitor_signals(embeddings, names, record.scoring)

    seed = s.calibration_seed
    calib = calibration_size_study(make_signals, queries, rows, n_runs=n_runs, seed=seed)
    window = window_size_study(
        make_signals, queries, rows, n_runs=n_runs, gradual_runs=gradual_runs, seed=seed
    )
    full = [next(r for r in calib if r["fraction"] >= 1.0)] + [
        r for r in window if r["window_size"] == 200
    ]
    for r in full:
        for name, frozen in record.thresholds.items():
            if not math.isclose(r["thresholds"][name], frozen, rel_tol=1e-9):
                raise RuntimeError(
                    f"{name}: full-data W=200 threshold {r['thresholds'][name]} != frozen {frozen}"
                )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"scoring": record.scoring, "calibration_size": calib, "window_size": window},
            indent=2,
            default=float,
        )
        + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(sensitivity_markdown(calib, window, record.scoring))
    typer.echo(report_md.read_text())


@app.command("length-matched")
def length_matched(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown table."),
    n_runs: int = typer.Option(200, help="Replay runs per fault."),
) -> None:
    """F1 and F2 on the questions where they answer as long as the healthy model (Stage 7f).

    Can the monitor see a real quality drop when answer length gives nothing
    away? Replayed against a FROZEN monitor, verified first.
    """
    from detect.calibrate import fit_signals
    from evaluate.lengthmatched import length_matched_markdown, run_length_matched
    from evaluate.pipeline import build_monitor_signals, calibration_rows
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ())
    calib_q, calib_rows = calibration_rows(queries, rows)
    signals = fit_signals(
        build_monitor_signals(embeddings, list(record.thresholds), record.scoring),
        calib_q,
        calib_rows,
    )
    results = run_length_matched(
        queries, rows, signals, record.thresholds, n_runs=n_runs, seed=s.calibration_seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"scoring": record.scoring, "results": results}, indent=2, default=float) + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(length_matched_markdown(results, record.scoring))
    typer.echo(report_md.read_text())


@app.command("one-slice")
def one_slice(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown tables."),
    n_runs: int = typer.Option(200, help="Replay runs per fault and slice."),
) -> None:
    """F1, F2 and F5 on one kind of question at a time, the rest healthy (Stage 7i).

    Replayed against a FROZEN monitor, verified first.
    """
    from detect.calibrate import fit_signals
    from evaluate.oneslice import (
        alias_one_slice,
        one_slice_conditions,
        one_slice_markdown,
        run_one_slice,
    )
    from evaluate.pipeline import build_monitor_signals, calibration_rows
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ())
    if embeddings is not None:
        embeddings = alias_one_slice(embeddings, one_slice_conditions(queries, rows)[2])
    calib_q, calib_rows = calibration_rows(queries, rows)
    signals = fit_signals(
        build_monitor_signals(embeddings, list(record.thresholds), record.scoring),
        calib_q,
        calib_rows,
    )
    results = run_one_slice(
        queries, rows, signals, record.thresholds, n_runs=n_runs, seed=s.calibration_seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"scoring": record.scoring, "results": results}, indent=2, default=float) + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(one_slice_markdown(results, record.scoring))
    typer.echo(report_md.read_text())


@app.command()
def sequential(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown tables."),
    n_runs: int = typer.Option(200, help="Replay runs per condition and schedule."),
) -> None:
    """CUSUM and Page-Hinkley next to the frozen thresholds (Stage 7j).

    Their settings come from healthy calib streams only; then all three run on
    the evaluation's and the gradual study's own streams. The record is
    verified first.
    """
    from corpus.splits import fault_subset
    from detect.calibrate import fit_signals
    from evaluate.gradual import GRADUAL_FAULTS, run_gradual
    from evaluate.pipeline import (
        DEFAULT_MIN_PREFIX,
        DEFAULT_REPLAY_WINDOWS,
        alias_embeddings,
        build_monitor_signals,
        calibration_rows,
        fault_conditions,
        replay_runs,
    )
    from evaluate.replay import run_replay
    from evaluate.sequential import (
        CALIB,
        CALIB_STREAMS,
        calibrate_sequential,
        calibration_traces,
        compare,
        pre_onset_offset,
        sequential_markdown,
    )
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ("F4", "F6"))
    calib_q, calib_rows = calibration_rows(queries, rows)
    if embeddings is not None:
        embeddings = alias_embeddings(embeddings, calib_rows, CALIB)
    signals = fit_signals(
        build_monitor_signals(embeddings, list(record.thresholds), record.scoring),
        calib_q,
        calib_rows,
    )
    calib = calibration_traces(
        calib_q,
        calib_rows,
        signals,
        record.thresholds,
        n_streams=CALIB_STREAMS,
        seed=s.calibration_seed + 1,
    )
    fitted = calibrate_sequential(calib, record.per_signal_alpha)

    healthy = [c for c in rows if c.condition == "healthy"]
    faults = fault_conditions(queries, rows)
    runs = replay_runs(
        list(faults), n_runs, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, s.calibration_seed
    )
    main_records = run_replay(
        runs, healthy, faults, fault_subset(queries), signals, record.thresholds
    )
    gradual_records = run_gradual(
        queries,
        healthy,
        {c: faults[c] for c in GRADUAL_FAULTS},
        signals,
        record.thresholds,
        n_runs=n_runs,
        seed=s.calibration_seed,
        keep_scores=True,
    )
    main = compare(main_records, record.thresholds, fitted, ["condition"])
    gradual = compare(gradual_records, record.thresholds, fitted, ["condition", "schedule"])
    fitted["pre_onset_offset_sd"] = pre_onset_offset(main_records, fitted)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"scoring": record.scoring, "fitted": fitted, "main": main, "gradual": gradual},
            indent=2,
            default=float,
        )
        + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(sequential_markdown(fitted, main, gradual, record.scoring))
    typer.echo(report_md.read_text())


@app.command("half-sample")
def half_sample(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown table."),
    n_runs: int = typer.Option(200, help="Replay runs per condition, on every half."),
) -> None:
    """Detection and false-alarm intervals over random halves of the questions (Stage 7l).

    Replayed against a FROZEN monitor, verified first.
    """
    from detect.calibrate import fit_signals
    from evaluate.halfsample import halfsample_markdown, run_halfsample
    from evaluate.pipeline import build_monitor_signals, calibration_rows
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ("F4", "F6"))
    calib_q, calib_rows = calibration_rows(queries, rows)
    signals = fit_signals(
        build_monitor_signals(embeddings, list(record.thresholds), record.scoring),
        calib_q,
        calib_rows,
    )
    results = run_halfsample(
        queries, rows, signals, record.thresholds, n_runs=n_runs, seed=s.calibration_seed
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scoring": record.scoring, **results}, indent=2) + "\n")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(halfsample_markdown(results, record.scoring))
    typer.echo(report_md.read_text())


@app.command("one-answer")
def one_answer(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown table."),
    n_runs: int = typer.Option(200, help="Replay runs per condition."),
) -> None:
    """A monitor that reads one answer per question instead of four (Stage 7m).

    Exploratory: re-calibrated from calib, nothing frozen. The record is
    verified first and gives the signal set and scoring.
    """
    from evaluate.pipeline import build_monitor_signals
    from evaluate.single import run_single, single_markdown
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ("F4", "F6"))

    def make_signals(names: Sequence[str]) -> dict:
        return build_monitor_signals(embeddings, list(names), record.scoring)

    results = run_single(
        queries,
        rows,
        make_signals,
        list(record.thresholds),
        n_runs=n_runs,
        seed=s.calibration_seed,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"scoring": record.scoring, **results}, indent=2, default=float) + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(single_markdown(results, record.scoring))
    typer.echo(report_md.read_text())


@app.command("second-model")
def second_model(
    completions: Path = typer.Option(..., help="All completions, the chosen H round included."),
    thresholds: Path = typer.Option(..., help="Frozen record giving the signals and scoring."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown table."),
    n_runs: int = typer.Option(200, help="Replay runs per condition."),
    deployed: str = typer.Option(
        "H2", help="H2 (gemma3:12b), H3 (qwen2.5:14b), H4 (llama3.1:8b) or H5 (phi4:14b)."
    ),
) -> None:
    """The method with another healthy model: gemma3, qwen2.5, llama3.1 or phi4 (7n-7q).

    Exploratory: calibrated afresh on that model's calib answers with the frozen
    record's signals and scoring; nothing is frozen. The record is verified first.
    """
    from evaluate.pipeline import alias_embeddings, build_monitor_signals, derive_f4, derive_f6
    from evaluate.secondmodel import (
        DEPLOYED,
        accuracy_table,
        second_model_markdown,
        second_model_rows,
        second_model_vectors,
        verdict,
    )
    from evaluate.sensitivity import calibrate_and_replay
    from settings import get_settings

    if deployed not in DEPLOYED:
        raise typer.BadParameter(f"deployed must be one of {sorted(DEPLOYED)}, got {deployed!r}")
    s = get_settings()
    queries, all_rows, record, embeddings = _frozen_inputs(completions, thresholds, ())
    rows = second_model_rows(queries, all_rows, deployed)
    healthy = [c for c in rows if c.condition == "healthy"]
    vectors = None
    if embeddings is not None:
        vectors = second_model_vectors(queries, embeddings, deployed)
        vectors = alias_embeddings(vectors, derive_f4(queries, healthy, s.corpus_seed), "F4")
        vectors = alias_embeddings(vectors, derive_f6(queries, healthy, s.corpus_seed), "F6")
    names = list(record.thresholds)
    result = calibrate_and_replay(
        lambda: build_monitor_signals(vectors, names, record.scoring),
        queries,
        rows,
        [q for q in queries if q.split == "calib"],
        n_runs=n_runs,
        seed=s.calibration_seed,
    )
    accuracy = accuracy_table(queries, rows, deployed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "deployed": deployed,
                "scoring": record.scoring,
                "accuracy": accuracy,
                "verdict": verdict(result, accuracy),
                **result,
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(second_model_markdown(result, accuracy, record.scoring, deployed))
    typer.echo(report_md.read_text())


@app.command("jsonx-check")
def jsonx_check(
    completions: Path = typer.Option(..., help="All completions."),
    report_md: Path = typer.Option(..., help="Markdown table."),
) -> None:
    """Score jsonx strictly vs priority-within-one, and which field each miss gets wrong.

    Diagnostic for the ambiguous answer key; every condition on the same
    fault-subset questions. The strict oracle and every report mean are unchanged.
    """
    from corpus.build import read_queries
    from corpus.splits import fault_subset
    from oracle.score import JSONX_EXACT_FIELDS, jsonx_field_check
    from provider.collect import read_completions
    from settings import get_settings

    queries = read_queries(get_settings().queries_path)
    ids = {q.query_id for q in fault_subset(queries)}
    rows = [c for c in read_completions(completions) if c.query_id in ids]
    conds = ["healthy", *sorted({c.condition for c in rows} - {"healthy"})]
    lines = [
        "# jsonx: strict key vs priority within one\n",
        "Same 240 fault-subset answers per condition (60 questions x 4). `lenient` accepts an "
        "answer whose only mismatch is a priority off by one. `wrong_*`: share of answers "
        "with that field wrong. Diagnostic; the oracle's strict rule is unchanged.\n",
        "| condition | strict | lenient | invalid JSON | "
        + " | ".join(f"wrong {f}" for f in JSONX_EXACT_FIELDS)
        + " |",
        "|---|---|---|---|" + "---|" * len(JSONX_EXACT_FIELDS),
    ]
    for cond in conds:
        r = jsonx_field_check(rows, queries, cond)
        lines.append(
            f"| {cond} | {100 * r['strict']:.1f}% | {100 * r['lenient']:.1f}% | {r['invalid']} | "
            + " | ".join(f"{100 * r[f'wrong_{f}']:.1f}%" for f in JSONX_EXACT_FIELDS)
            + " |"
        )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines) + "\n")
    typer.echo(report_md.read_text())


@app.command("leakage-check")
def leakage_check_cmd(
    completions: Path = typer.Option(..., help="All completions."),
    report_md: Path = typer.Option(..., help="Markdown report."),
) -> None:
    """Check the real splits and answers for leakage: ids, prompts, source material, answers."""
    from corpus.build import read_queries
    from corpus.leakage import leakage_check, leakage_markdown
    from provider.collect import read_completions
    from settings import get_settings

    r = leakage_check(read_queries(get_settings().queries_path), read_completions(completions))
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(leakage_markdown(r))
    typer.echo(report_md.read_text())
    if r["shared_prompts"] or r["answers"]["misplaced"]:
        raise typer.Exit(1)


@app.command("pilot-f8")
def pilot_f8(
    completions: Path = typer.Option(..., help="Completions holding the healthy round."),
    out: Path = typer.Option(..., help="Pilot results JSON."),
    per_slice: int = typer.Option(20, help="Calib queries per slice."),
    concurrency: int = typer.Option(4, help="Requests in flight."),
) -> None:
    """Pick F8's prompt by the rule fixed in config/extension_prompts.yaml.

    Answers a small CALIB sample with each candidate prompt (one sample, the
    healthy round's sample-0 seeds) and compares accuracy and answer length
    with the healthy answers to the same questions. Never touches the fault
    split, and no monitor scores anything here.
    """
    import numpy as np

    from corpus.build import read_queries
    from oracle.score import score_condition
    from provider.collect import collect_results, read_completions
    from provider.expand import expand_queries
    from settings import get_settings, load_extension_prompts
    from signals.length import completion_length

    s = get_settings()
    scorable = ("banking77", "gsm8k", "squad2", "triviaqa")
    queries = read_queries(s.queries_path)
    rng = np.random.default_rng(s.calibration_seed)
    picked = []
    for name in (*scorable, "xsum", "eli5"):
        pool = sorted(
            (q for q in queries if q.slice == name and q.split == "calib"),
            key=lambda q: q.query_id,
        )
        picked += [pool[int(i)] for i in sorted(rng.choice(len(pool), per_slice, replace=False))]
    ids = {q.query_id for q in picked}
    healthy = [
        c
        for c in read_completions(completions)
        if c.condition == "healthy" and c.sample_idx == 0 and c.query_id in ids
    ]

    def measure(rows: list) -> dict:
        acc = {r.slice: r.accuracy for r in score_condition(rows, queries, rows[0].condition)}
        return {
            "accuracy_by_slice": {k: acc[k] for k in scorable},
            "accuracy": float(np.mean([acc[k] for k in scorable])),
            "mean_length": float(np.mean([completion_length(c) for c in rows])),
        }

    ext = load_extension_prompts()
    resolved = _resolve_role("healthy", "local")
    results = {"healthy": measure(healthy), "candidates": {}, "rule_version": ext["version"]}
    chosen = ""
    for name, prompt in ext["f8_candidates"].items():
        rows = expand_queries(
            picked,
            1,
            resolved.model_id,
            "F8",
            "p001",
            prompt,
            s.seed_base,
            s.temperature,
            s.max_tokens,
        )
        raw = out.with_name(f"pilot_f8_{name}.jsonl")
        _answer_rows(rows, raw, "local", concurrency)
        got, report = collect_results(
            raw,
            {r["custom_id"] for r in rows},
            resolved.model_id,
            resolved.digest,
            resolved.digest_source,
            datetime.now(UTC),
            seeds={r["custom_id"]: r["body"]["seed"] for r in rows},
        )
        report.raise_if_dirty()
        m = measure(got)
        m["accuracy_drop_points"] = 100 * (results["healthy"]["accuracy"] - m["accuracy"])
        m["length_ratio"] = m["mean_length"] / results["healthy"]["mean_length"]
        m["qualifies"] = m["accuracy_drop_points"] >= 5 and abs(m["length_ratio"] - 1) <= 0.10
        results["candidates"][name] = m
        if m["qualifies"] and not chosen:
            chosen = name  # candidates are listed in preference order
        typer.echo(
            f"{name:14} drop {m['accuracy_drop_points']:+.1f} pts, length x{m['length_ratio']:.3f}"
            f" -> {'qualifies' if m['qualifies'] else 'does not qualify'}"
        )
    results["chosen"] = chosen
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    typer.echo(f"chosen: {chosen or 'NONE - no length-matched fault found'}")


@app.command("remote-check")
def remote_check() -> None:
    """Check every pinned remote deployment is served, and show the spend so far (free).

    Lists each deployment in config/remote.yaml with its endpoint and the
    router's current price, and the total recorded on disk against the cap.
    Sends no chat request, so it costs nothing.
    """
    from provider.remote import RemoteError, spent_usd
    from settings import load_remote_config

    cfg = load_remote_config()
    spent = spent_usd(_rounds_dir())
    ok = True
    for name in cfg["deployments"]:
        client = _remote_client(name, spent)
        try:
            pin = client.resolve(name)
            typer.echo(f"ok    {name:<16} {pin.digest}  ({client.dep['quantization']})")
        except RemoteError as exc:
            ok = False
            typer.echo(f"FAIL  {name:<16} {exc}")
        finally:
            client.close()
    typer.echo(f"spent ${spent:.4f} of the ${cfg['budget_usd']:.2f} cap")
    if not ok:
        raise typer.Exit(1)


@app.command("remote-pilot")
def remote_pilot(
    deployment: str = typer.Option(..., help="A deployment in config/remote.yaml."),
    per_slice: int = typer.Option(2, help="Fault-subset questions per slice, one sample each."),
    round_id: str = typer.Option("apipilot1", help="Round id; the checkpoint is kept."),
) -> None:
    """A few real answers from one pinned deployment, to measure tokens and cost (SPENDS).

    Healthy prompt and seeds, one sample per question, a handful of questions
    per slice. Checks every answer came from the pinned host without thinking,
    and extrapolates the cost of a full round. Nothing enters completions.parquet.
    """
    from collections import defaultdict

    from corpus.build import read_queries
    from corpus.splits import fault_subset
    from provider.expand import expand_queries
    from provider.local import run_rows
    from provider.remote import spent_usd
    from settings import get_settings, load_remote_config, system_prompt_for_condition

    s = get_settings()
    by_slice: dict[str, list] = defaultdict(list)
    for q in sorted(fault_subset(read_queries(s.queries_path)), key=lambda q: q.query_id):
        if len(by_slice[q.slice]) < per_slice:
            by_slice[q.slice].append(q)
    queries = [q for qs in by_slice.values() for q in qs]
    client = _remote_client(deployment, spent_usd(_rounds_dir()))
    pin = client.resolve(deployment)
    rows = expand_queries(
        queries,
        1,
        pin.model_id,
        "healthy",
        round_id,
        system_prompt_for_condition("healthy"),
        s.seed_base,
        s.temperature,
        s.max_tokens,
    )
    checkpoint = _rounds_dir() / round_id / "results.jsonl"
    try:
        run_rows(client, rows, checkpoint, concurrency=4)  # type: ignore[arg-type]
    finally:
        client.close()
    lines = [json.loads(line) for line in checkpoint.read_text(encoding="utf-8").splitlines()]
    usage = [(ln["response"] or {}).get("usage") or {} for ln in lines]
    cost = sum(float(u.get("cost") or 0.0) for u in usage)
    tok_in = sum(int(u.get("prompt_tokens") or 0) for u in usage)
    tok_out = sum(int(u.get("completion_tokens") or 0) for u in usage)
    n = len(lines)
    typer.echo(f"{deployment}: {n} answers from {pin.digest}")
    typer.echo(
        f"  tokens in {tok_in} (mean {tok_in / n:.0f}), out {tok_out} (mean {tok_out / n:.0f})"
    )
    typer.echo(f"  cost ${cost:.5f} (mean ${cost / n:.6f} per answer)")
    for answers in (6_000, 26_000):
        typer.echo(f"  -> {answers:,} answers at this mean: ~${cost / n * answers:.2f}")
    cap = load_remote_config()["budget_usd"]
    typer.echo(f"  spent so far ${spent_usd(_rounds_dir()):.4f} of the ${cap:.2f} cap")


@app.command()
def judge(
    completions: Path = typer.Option(..., help="Completions holding healthy and the faults."),
    out: Path = typer.Option(..., help="Judge score store (parquet)."),
    concurrency: int = typer.Option(4, help="Requests in flight; match OLLAMA_NUM_PARALLEL."),
) -> None:
    """LLM-as-judge round (Stage 7m): a second model grades every answer 1-10.

    Grades every sample of healthy, F1, F2, F3, F5 and F7 with the model, prompt
    and parsing pre-registered in config/judge.yaml. Checkpointed per row like a
    generation round: rerun the same command to resume. Writes the score store
    and a summary JSON next to it.
    """
    from corpus.build import read_queries
    from evaluate.judge import (
        JUDGED_CONDITIONS,
        build_judge_rows,
        collect_judge_scores,
        judge_summary,
        write_judge_scores,
    )
    from provider.collect import read_completions
    from settings import get_settings, load_judge_config

    s = get_settings()
    cfg = load_judge_config()
    resolved = _resolve_role(cfg["role"], "local")
    queries = read_queries(s.queries_path)
    graded = [c for c in read_completions(completions) if c.condition in JUDGED_CONDITIONS]
    rows = build_judge_rows(queries, graded, resolved.model_id)
    checkpoint = _rounds_dir() / cfg["round_id"] / "results.jsonl"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    typer.echo(f"{len(rows)} answers to grade with {resolved.model_id}")
    _answer_rows(rows, checkpoint, "local", concurrency)
    scores, replies = collect_judge_scores(
        checkpoint, rows, resolved.model_id, resolved.digest, resolved.digest_source
    )
    write_judge_scores(scores, replies, out)
    summary = {
        "judge_model": resolved.model_id,
        "judge_digest": resolved.digest,
        "judge_config_version": cfg["version"],
        "by_condition": judge_summary(scores),
    }
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    for cond, r in summary["by_condition"].items():
        typer.echo(
            f"{cond:8} graded {r['graded']:>6}  mean {r['mean_score']:.3f}  "
            f"unscored {r['unscored']}"
        )


@app.command()
def ablation(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown tables."),
    n_runs: int = typer.Option(200, help="Replay runs per condition, as in `evaluate`."),
    judge_scores: Path = typer.Option(
        None, help="The `judge` round's score store; adds the LLM-judge baseline."
    ),
) -> None:
    """Signal ablation and simple baselines on `evaluate`'s own streams (Stage 7c).

    Each signal removed in turn, each alone, a mean-length baseline and the
    uncorrected monitor - all at the same family-wise budget except that last
    one. Refuses unless the bootstrap reproduces the frozen thresholds. With
    --judge-scores, the LLM-judge baseline (Stage 7m) is one more row.
    """
    from evaluate.ablation import ablation_markdown, run_ablation
    from evaluate.pipeline import build_monitor_signals, derive_f4, derive_f6
    from settings import get_settings, load_judge_config

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, ("F4", "F6"))
    signals = build_monitor_signals(embeddings, list(record.thresholds), record.scoring)
    judge = None
    if judge_scores is not None:
        from evaluate.judge import alias_judge_scores, build_judge_signal, read_judge_scores

        scores = read_judge_scores(judge_scores)
        healthy = [c for c in rows if c.condition == "healthy"]
        for name, derive in (("F4", derive_f4), ("F6", derive_f6)):
            scores = alias_judge_scores(scores, derive(queries, healthy, s.corpus_seed), name)
        judge = build_judge_signal(scores, record.scoring)
    results = run_ablation(
        queries,
        rows,
        signals,
        record.thresholds,
        n_runs=n_runs,
        seed=s.calibration_seed,
        embeddings=embeddings,
        judge=judge,
    )
    results["scoring"] = record.scoring
    if judge is not None:
        results["judge_config_version"] = load_judge_config()["version"]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(ablation_markdown(results, record.scoring))
    typer.echo(report_md.read_text())


@app.command()
def mixtures(
    completions: Path = typer.Option(..., help="Healthy completions (faults are ignored)."),
    thresholds: Path = typer.Option(..., help="A frozen thresholds record (v1, v2 or v3)."),
    out: Path = typer.Option(..., help="Results JSON."),
    report_md: Path = typer.Option(..., help="Markdown summary."),
    n_runs: int = typer.Option(100, help="Replays per mixture."),
) -> None:
    """False alarms over many random harmless query shifts (Stage 7d).

    50 random query mixtures, healthy answers throughout, replayed against a
    FROZEN monitor; every alarm after the onset is a false alarm.
    """
    from detect.calibrate import fit_signals
    from evaluate.mixtures import (
        MIX,
        draw_mixtures,
        mixtures_markdown,
        run_mixtures,
        summarise_mixtures,
    )
    from evaluate.pipeline import build_monitor_signals, calibration_rows
    from settings import get_settings

    s = get_settings()
    queries, rows, record, embeddings = _frozen_inputs(completions, thresholds, (MIX,))
    names = list(record.thresholds)
    calib_q, calib_rows = calibration_rows(queries, rows)
    signals = fit_signals(
        build_monitor_signals(embeddings, names, record.scoring), calib_q, calib_rows
    )
    drawn = draw_mixtures(queries, s.calibration_seed)
    healthy = [c for c in rows if c.condition == "healthy"]
    records = run_mixtures(
        queries,
        healthy,
        signals,
        record.thresholds,
        drawn,
        n_runs=n_runs,
        seed=s.calibration_seed,
    )
    summary = summarise_mixtures(records, drawn, names)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"scoring": record.scoring, "summary": summary, "records": records}, indent=2)
        + "\n"
    )
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text(mixtures_markdown(summary, record.scoring))
    typer.echo(report_md.read_text())


# ── helpers ────────────────────────────────────────────────────────────────


def _frozen_inputs(
    completions: Path, thresholds: Path, aliases: Sequence[str]
) -> tuple[list, list, Any, dict | None]:
    """Queries, rows, the VERIFIED frozen record, and embeddings with derived conditions aliased.

    Refuses to continue on any mismatch between the record and its calibration
    rows, exactly as `evaluate` does. Each name in `aliases` gets the healthy
    vectors re-keyed under it: F4 and F6 for their own mixtures, MIX for all
    healthy rows.
    """
    from corpus.build import read_queries
    from detect import freeze
    from evaluate.pipeline import alias_embeddings, calibration_rows, derive_f4, derive_f6
    from provider.collect import read_completions, read_embeddings
    from settings import get_settings

    s = get_settings()
    queries = read_queries(s.queries_path)
    rows = read_completions(completions)
    _calib_q, calib_rows = calibration_rows(queries, rows)
    record = freeze.verify(thresholds, calib_rows)
    embeddings = None
    if "embedding_shift" in record.thresholds:
        embeddings = read_embeddings(_embeddings_path())
        healthy = [c for c in rows if c.condition == "healthy"]
        derive = {"F4": derive_f4, "F6": derive_f6}
        for name in aliases:
            covered = derive[name](queries, healthy, s.corpus_seed) if name in derive else healthy
            embeddings = alias_embeddings(embeddings, covered, name)
    return queries, rows, record, embeddings


def _client() -> BatchClient:
    from provider.client import BatchClient
    from settings import get_settings

    s = get_settings()
    return BatchClient(s.base_url, s.api_key, s.request_timeout_s)


def _local_client(timeout: float | None = None) -> LocalClient:
    from provider.local import DEFAULT_TIMEOUT_S, LocalClient
    from settings import get_settings

    s = get_settings()
    return LocalClient(s.base_url, s.api_key, timeout or DEFAULT_TIMEOUT_S)


def _resolve_role(role: str, backend: str) -> ResolvedModel:
    """Resolve a role on the chosen backend, checking the local build against the pin (C9)."""
    from provider.catalogue import RoleResolutionError, resolve_roles
    from settings import load_model_roles

    if backend not in ("local", "batch"):
        raise typer.BadParameter(f"backend must be 'local' or 'batch', got {backend!r}")
    if backend == "batch":
        return resolve_roles(_client(), [role])[role]

    client = _local_client()
    log.info("local.server", ollama_version=client.version())
    resolved = resolve_roles(client, [role])[role]  # type: ignore[arg-type]
    pinned = load_model_roles()[role].get("library_digest")
    if pinned and not resolved.digest.startswith(str(pinned)):
        raise RoleResolutionError(
            f"{resolved.model_id}: the local server has build {resolved.digest[:12]}, but "
            f"config/models.yaml pins {pinned}. The library build changed - check it before "
            "running, then update the pin deliberately."
        )
    return resolved


def _round_rows(condition: str, round_id: str) -> tuple[list[dict], str]:
    """The rows of one generation round, and the role that serves it.

    healthy runs the whole corpus; F1, F2, F3, F5 and F7 the ONE shared fault
    subset (C8); H2 the calib and heldout questions; H3, H4 and H5 the whole
    corpus, like healthy. F4 and F6 have no round -
    they are derived from healthy completions.
    """
    from corpus.build import read_queries
    from corpus.splits import fault_subset
    from provider.expand import expand_queries
    from settings import (
        CONDITION_ROLES,
        F7_SEED_BASE,
        FIFTH_HEALTHY,
        FOURTH_HEALTHY,
        SECOND_HEALTHY,
        THIRD_HEALTHY,
        get_settings,
        model_id_for_role,
        system_prompt_for_condition,
    )

    s = get_settings()
    if condition in ("F4", "F6"):
        raise typer.BadParameter(
            f"{condition} has no generation round - it is derived from healthy rows"
        )
    if condition not in CONDITION_ROLES:
        raise typer.BadParameter(
            f"unknown condition {condition!r}; one of {sorted(CONDITION_ROLES)}"
        )
    role = CONDITION_ROLES[condition]
    queries = read_queries(s.queries_path)
    if condition in ("healthy", THIRD_HEALTHY, FOURTH_HEALTHY, FIFTH_HEALTHY):
        selected = queries
    elif condition == SECOND_HEALTHY:
        # The fault questions are F5's round already: same model, prompt and seeds.
        selected = [q for q in queries if q.split != "fault"]
    else:
        selected = fault_subset(queries)
    rows = expand_queries(
        selected,
        s.k,
        model_id_for_role(role),
        condition,
        round_id,
        system_prompt_for_condition(condition),
        F7_SEED_BASE if condition == "F7" else s.seed_base,
        s.temperature,
        s.max_tokens,
    )
    return rows, role


def _answer_rows(rows: list[dict], out: Path, backend: str, concurrency: int) -> None:
    """Answer rows on either backend, writing batch-output-shaped lines to `out`."""
    if backend == "local":
        from provider.local import run_rows

        client = _local_client()
        # Ollama reuses the KV cache of a prompt it has just processed. The
        # cold pass and the cached pass differ in the last bits, and over a
        # few hundred tokens one sampled token flips: a fixed seed reproduces
        # exactly WITHIN each cache state, not across them (measured on the
        # 26B: cold always gives one text, warm always another). One unscored
        # request first puts every scored row in the same, warm, state.
        client.post_row(rows[0])
        run_rows(client, rows, out, concurrency)
        return
    from provider.expand import write_jsonl
    from provider.submit import poll_until_complete, submit_batch
    from settings import CHAT_URL, get_settings

    s = get_settings()
    client = _client()
    path = out.with_name("probe.jsonl")
    write_jsonl(rows, path)
    submitted = submit_batch(client, path, CHAT_URL, "probe", "probe")
    batch = poll_until_complete(client, submitted.batch_id, s.poll_interval_s, s.poll_timeout_s)
    _download_batch(client, batch, out)


def _resolve_model(model_id: str) -> ResolvedModel:
    """Resolve a model id against the live catalogue, by the role that maps to it."""
    from provider.catalogue import RoleResolutionError, resolve_roles
    from settings import load_model_roles

    roles = [r for r, e in load_model_roles().items() if e["id"] == model_id]
    if not roles:
        raise RoleResolutionError(f"{model_id} is not configured in config/models.yaml")
    return resolve_roles(_client(), roles[:1])[roles[0]]


def _remote_client(name: str, already_spent: float = 0.0) -> Any:
    """A RemoteClient pinned to one deployment. Needs DRIFTWATCH_REMOTE_* in .env."""
    from provider.remote import RemoteClient, deployment
    from settings import get_settings, load_remote_config

    s = get_settings()
    if not s.remote_base_url or not s.remote_api_key:
        raise typer.BadParameter(
            "set DRIFTWATCH_REMOTE_BASE_URL and DRIFTWATCH_REMOTE_API_KEY in .env first"
        )
    return RemoteClient(
        s.remote_base_url,
        s.remote_api_key,
        deployment(name),
        budget_usd=load_remote_config()["budget_usd"],
        already_spent_usd=already_spent,
    )


def _rounds_dir() -> Path:
    from settings import get_settings

    return get_settings().data_dir / "rounds"


def _manifest_path(round_id: str) -> Path:
    return _rounds_dir() / round_id / "manifest.json"


def _completions_path() -> Path:
    from settings import get_settings

    return get_settings().completions_path


def _embeddings_path() -> Path:
    from settings import get_settings

    return get_settings().data_dir / "embeddings.parquet"


def _now() -> datetime:
    return datetime.now(UTC)


def _download_batch(client: BatchClient, batch: dict, dest: Path) -> None:
    """Output file + error file, concatenated (errors are rows too - C6)."""
    parts = []
    for key in ("output_file_id", "error_file_id"):
        if batch.get(key):
            part = dest.with_name(f"{dest.name}.{key}")
            client.download_file(batch[key], part)
            parts.append(part.read_text())
    dest.write_text("".join(p if p.endswith("\n") or not p else p + "\n" for p in parts))


def _calibration_inputs(completions: Path) -> tuple[list, list]:
    from corpus.build import read_queries
    from evaluate.pipeline import calibration_rows
    from provider.collect import read_completions
    from settings import get_settings

    queries = read_queries(get_settings().queries_path)
    return calibration_rows(queries, read_completions(completions))


def _enabled(no_embeddings: bool, without: Sequence[str] = ()) -> list[str]:
    """Signals to calibrate. The frozen record carries this list; evaluate reads it back."""
    from settings import SIGNAL_NAMES

    unknown = set(without) - set(SIGNAL_NAMES)
    if unknown:
        raise typer.BadParameter(f"unknown signal(s) {sorted(unknown)}; known: {SIGNAL_NAMES}")
    dropped = set(without) | ({"embedding_shift"} if no_embeddings else set())
    return [n for n in SIGNAL_NAMES if n not in dropped]


def _embeddings_or_none(no_embeddings: bool) -> dict | None:
    from provider.collect import read_embeddings

    if no_embeddings:
        return None
    path = _embeddings_path()
    if not path.exists():
        raise typer.BadParameter(
            f"no embedding store at {path}: run the embedding round, or pass --no-embeddings"
        )
    return read_embeddings(path)


def _synthetic_evaluation(n_runs: int) -> dict:
    """Stage 7 gate on synthetic data: calibrate in memory (NOT frozen) and replay.

    Fault strengths are fixed here and labelled as synthetic in the output; the
    numbers say the PIPELINE works, nothing about real models.
    """
    from corpus.build import read_queries
    from corpus.splits import fault_subset
    from detect.calibrate import calibrate_with_report, sidak_alpha
    from evaluate.pipeline import calibration_rows, evaluate_run
    from settings import get_settings
    from signals.registry import build_signals
    from synthetic.generate import DegradationKnob, generate_completions

    s = get_settings()
    queries = read_queries(s.queries_path)
    fault = fault_subset(queries)
    rows = generate_completions(queries, DegradationKnob(), s.k, "healthy", s.seed_base)
    rows += generate_completions(fault, DegradationKnob.uniform(0.6), s.k, "F1", s.seed_base)
    rows += generate_completions(fault, DegradationKnob.uniform(0.25), s.k, "F2", s.seed_base)
    rows += generate_completions(
        fault, DegradationKnob(length=0.3, refusal=0.3), s.k, "F3", s.seed_base
    )
    calib_q, calib_rows = calibration_rows(queries, rows)
    signals = build_signals()  # S7 uses its offline lexical stand-in here, by design
    thresholds, _fits = calibrate_with_report(
        signals, calib_q, calib_rows, s.false_alarm_budget, s.bootstrap_windows, s.calibration_seed
    )
    return evaluate_run(
        queries,
        rows,
        signals,
        thresholds,
        per_signal_alpha=sidak_alpha(s.false_alarm_budget, len(signals)),
        n_runs=n_runs,
        seed=s.calibration_seed,
        mode="synthetic",
    )


if __name__ == "__main__":
    app()

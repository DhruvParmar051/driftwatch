"""Command-line interface. Every command wired to the library.

Commands run in the Stage order of the build plan. Each one is thin: parse
arguments, configure logging, call into the library, report. Put no logic here
that a test would want to reach directly.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from log import configure_logging, get_logger
from version import __version__

if TYPE_CHECKING:
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
    this is where that shows up.

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
    condition: str = typer.Option(..., help="healthy, F1, F2, F3 or F5."),
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
    signals = build_monitor_signals(_embeddings_or_none(no_embeddings), _enabled(no_embeddings))
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
    record = freeze.write(
        s.thresholds_path,
        proposal["thresholds"],
        rows,
        proposal["family_wise_budget"],
        proposal["per_signal_alpha"],
        s.k,
        s.window_size,
        s.window_stride,
        proposal["bootstrap_windows"],
        proposal["calibration_seed"],
    )
    typer.echo(f"FROZEN at commit {record.git_commit}: {s.thresholds_path}")


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
        signals = fit_signals(build_monitor_signals(embeddings, names), calib_q, calib_rows)
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
        out = s.data_dir / "results" / "evaluation.json"
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


# ── helpers ────────────────────────────────────────────────────────────────


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

    healthy runs the whole corpus; F1, F2, F3 and F5 the ONE shared fault
    subset (C8). F4 has no round - it is derived from healthy completions.
    """
    from corpus.build import read_queries
    from corpus.splits import fault_subset
    from provider.expand import expand_queries
    from settings import (
        CONDITION_ROLES,
        get_settings,
        model_id_for_role,
        system_prompt_for_condition,
    )

    s = get_settings()
    if condition == "F4":
        raise typer.BadParameter("F4 has no generation round - it is derived from healthy rows")
    if condition not in CONDITION_ROLES:
        raise typer.BadParameter(
            f"unknown condition {condition!r}; one of {sorted(CONDITION_ROLES)}"
        )
    role = CONDITION_ROLES[condition]
    queries = read_queries(s.queries_path)
    selected = queries if condition == "healthy" else fault_subset(queries)
    rows = expand_queries(
        selected,
        s.k,
        model_id_for_role(role),
        condition,
        round_id,
        system_prompt_for_condition(condition),
        s.seed_base,
        s.temperature,
        s.max_tokens,
    )
    return rows, role


def _answer_rows(rows: list[dict], out: Path, backend: str, concurrency: int) -> None:
    """Answer rows on either backend, writing batch-output-shaped lines to `out`."""
    if backend == "local":
        from provider.local import run_rows

        run_rows(_local_client(), rows, out, concurrency)
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


def _enabled(no_embeddings: bool) -> list[str]:
    from settings import SIGNAL_NAMES

    return [n for n in SIGNAL_NAMES if not (no_embeddings and n == "embedding_shift")]


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

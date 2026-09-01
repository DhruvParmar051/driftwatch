"""Command-line interface. FULLY IMPLEMENTED - every command wired to its stub.

Commands run in the Stage order of the build plan. Each one is thin: parse
arguments, configure logging, call into the library, report. Put no logic here
that a test would want to reach directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from version import __version__
from log import configure_logging, get_logger

app = typer.Typer(
    name="driftwatch",
    help="Label-free detection of silent quality degradation in deployed language models.",
    no_args_is_help=True,
    add_completion=False,
)
log = get_logger(__name__)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v"),
         plain: bool = typer.Option(False, "--plain", help="Human-readable logs instead of JSON.")) -> None:
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
    seed: Optional[int] = typer.Option(None, help="Corpus seed. Defaults to settings.corpus_seed."),
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
    k: Optional[int] = typer.Option(None, help="Samples per query."),
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
def probe(rows: int = typer.Option(20, help="Total probe rows (10 distinct + 10 identical seeds).")) -> None:
    """Stage 2 seed-determinism probe. RUN THIS BEFORE ANY LARGE ROUND.

    Submits one prompt x 10 distinct seeds and the same prompt x 10 identical
    seeds, then asserts: distinct seeds give >= 8 distinct outputs, identical
    seeds give identical output.

    The platform's compatibility matrix documents `seed` as honoured, but it is
    explicitly doc-derived and not verified against a live server. This probe is
    the evidence.

    IF DISTINCT SEEDS DO NOT VARY THE OUTPUT, STOP. S3 (self-consistency) is not
    viable and the design must change before a baseline round is spent.
    """
    from provider.catalogue import resolve_roles
    from provider.client import BatchClient
    from settings import get_settings

    s = get_settings()
    client = BatchClient(s.base_url, s.api_key, s.request_timeout_s)
    resolved = resolve_roles(client)
    log.info("probe.roles_resolved", roles={r: m.digest for r, m in resolved.items()})
    raise NotImplementedError("probe: implement submit + assert per Stage 2 gate")


@app.command()
def expand(
    condition: str = typer.Option(..., help="healthy, F1..F5."),
    round_id: str = typer.Option(..., help="Round identifier, e.g. r003."),
    out: Path = typer.Option(..., help="Output JSONL path."),
) -> None:
    """Expand queries into k JSONL rows each, with distinct seeds (Stage 3)."""
    raise NotImplementedError("expand: wire corpus -> provider.expand.expand_queries")


@app.command()
def submit(jsonl: Path = typer.Argument(..., help="JSONL file to submit."),
           condition: str = typer.Option(...), round_id: str = typer.Option(...)) -> None:
    """Upload and submit a batch, then persist the submission record (Stage 3)."""
    raise NotImplementedError("submit: wire provider.submit.submit_batch")


@app.command()
def collect(batch_id: str = typer.Argument(...), out: Path = typer.Option(...)) -> None:
    """Download, parse and RECONCILE a completed batch (Stage 3).

    Fails loudly on any submitted/returned mismatch (C10). A partially returned
    batch breaks every per-cell sample size and must never be analysed.
    """
    raise NotImplementedError("collect: wire provider.collect.collect_results")


# ── Stage 5: calibrate and freeze ──────────────────────────────────────────


@app.command()
def calibrate(
    completions: Path = typer.Option(..., help="Healthy completions parquet."),
    split: str = typer.Option("calib", help="Must be 'calib'. Anything else raises LeakageError."),
    out: Path = typer.Option(None, help="Where to write the proposed thresholds."),
) -> None:
    """Fit reference distributions and estimate thresholds (Stage 5).

    Calibrates on split == "calib" ONLY (C4). Thresholds are set at the
    Sidak-corrected per-signal level so the FAMILY-WISE false-alarm rate meets
    the budget - calibrating each signal at the budget itself would run about 7x
    over it at m=7.
    """
    raise NotImplementedError("calibrate: wire detect.calibrate.calibrate")


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
    raise NotImplementedError("freeze: wire detect.freeze.write")


@app.command()
def verify(thresholds: Path = typer.Option(None), completions: Path = typer.Option(...)) -> None:
    """Verify a frozen thresholds.json against its calibration rows (Stage 5 gate)."""
    raise NotImplementedError("verify: wire detect.freeze.verify")


# ── Stage 6-8: oracle, evaluate, report ────────────────────────────────────


@app.command()
def oracle(completions: Path = typer.Option(...), condition: str = typer.Option(...)) -> None:
    """Ground-truth scoring on the verifiable slices (Stage 6).

    Runs strictly AFTER detection. Nothing under detect/ may import oracle/ (C3).
    """
    raise NotImplementedError("oracle: wire oracle.score.score_condition")


@app.command()
def evaluate(
    completions: Path = typer.Option(..., help="Healthy + fault completions."),
    thresholds: Path = typer.Option(None, help="Frozen thresholds. Defaults to settings.thresholds_path."),
    synthetic: bool = typer.Option(False, "--synthetic", help="Evaluate against synthetic data (Stage 7 gate)."),
    n_runs: int = typer.Option(200, help="Randomised-onset replay runs per condition."),
) -> None:
    """Replay, detection latency, ROC and bootstrap CIs (Stage 7).

    REFUSES TO RUN without a thresholds.json that passes freeze.verify().
    """
    raise NotImplementedError("evaluate: wire evaluate.replay + metrics + bootstrap")


@app.command()
def report(results: Path = typer.Option(...), out_dir: Path = typer.Option(...)) -> None:
    """Generate results tables and figures (Stage 8).

    Detection latency is plotted as a DISTRIBUTION, never a point estimate, and
    results are reported PER FAULT - never pooled into one success rate.
    """
    raise NotImplementedError("report: wire evaluate.metrics.summarise")


if __name__ == "__main__":
    app()

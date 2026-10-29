"""Usage: DRIFTWATCH_STUDY=chat uv run python scripts/chat_topics.py.

Topic shifts on real traffic without task labels (Stage 7u extension). Every
choice below was fixed and committed before this script was first run.

Real traffic carries no task family, so the per-slice monitor (v3) cannot be
built and the chat study had no topic-shift test. Here topics are found without
labels: the chat prompts are embedded (nomic-embed-text v1.5, the S7 model) and
clustered by k-means (K = 6, seed = corpus seed, fitted on calib prompts only).
Clusters are ranked by the median length of the healthy model's answers on calib
questions, and two harmless shifts are drawn from calib | fault questions, like
F4 and F6, never heldout:

  F4  900 questions, 70% from the three long-answer clusters, 30% from the rest
  F6  900 questions, 70% from the three short-answer clusters, 30% from the rest

within each half in proportion to cluster size. Their rows are the healthy
model's own answers, so nothing about the model changes. Replayed like every
other condition (200 streams, the frozen replay plan) under: the frozen chat
monitor (one slice, so pooled), the paired `rounds` monitor (committed record),
and the Model Equality Testing baseline. Results go to
data/chat/results/topics.json and data/report_paired/chat_topics.md.
"""

import json
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import evaluate.paired as paired
from cli import _paired_inputs
from corpus.build import read_queries
from detect import freeze
from detect.calibrate import fit_signals
from evaluate.metrics import wilson_interval
from evaluate.pipeline import (
    DEFAULT_MIN_PREFIX,
    DEFAULT_REPLAY_WINDOWS,
    HEALTHY,
    alias_embeddings,
    build_monitor_signals,
    calibration_rows,
    replay_runs,
)
from evaluate.replay import run_replay
from provider.collect import read_completions, read_embeddings
from settings import DATA_DIR, get_settings

K = 6
SIZE = 900
SHARE = 0.7
ROOT = DATA_DIR / "chat"


def embed_prompts(queries: list) -> np.ndarray:
    """Prompt vectors, cached; the S7 model with its document prefix."""
    cache = ROOT / "prompt_embeddings.npy"
    if cache.exists():
        return np.load(cache)
    out = []
    with httpx.Client(timeout=120) as client:
        for i in range(0, len(queries), 64):
            batch = [f"search_document: {q.prompt}" for q in queries[i : i + 64]]
            r = client.post(
                "http://localhost:11434/v1/embeddings",
                json={"model": "nomic-embed-text:v1.5", "input": batch},
            )
            r.raise_for_status()
            out.extend(d["embedding"] for d in r.json()["data"])
    vecs = np.asarray(out, dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(cache, vecs)
    return vecs


def kmeans(x: np.ndarray, k: int, seed: int, iters: int = 100) -> np.ndarray:
    """Plain Lloyd's k-means (k-means++ start); returns the centres."""
    rng = np.random.default_rng(seed)
    centres = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min([((x - c) ** 2).sum(1) for c in centres], axis=0)
        centres.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(centres)
    for _ in range(iters):
        label = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(2), axis=1)
        new = np.array([x[label == j].mean(0) if (label == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c


def draw(pool: dict[int, list], favoured: list[int], seed: int, stream: int) -> list:
    """SIZE questions: SHARE from the favoured clusters, the rest from the others."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, stream]))
    picked = []
    for group, n in ((favoured, round(SIZE * SHARE)), (sorted(set(pool) - set(favoured)), 0)):
        if not n:
            n = SIZE - len(picked)
        sizes = np.array([len(pool[j]) for j in group], dtype=float)
        quota = np.floor(n * sizes / sizes.sum()).astype(int)
        quota[: n - quota.sum()] += 1
        for j, m in zip(group, quota, strict=True):
            cands = sorted(pool[j], key=lambda q: q.query_id)
            picked.extend(cands[int(i)] for i in sorted(rng.choice(len(cands), m, replace=False)))
    return picked


def main() -> None:
    """Cluster, draw the two shifts and replay them under the three monitors."""
    s = get_settings()
    queries = read_queries(ROOT / "queries.jsonl")
    rows = read_completions(ROOT / "completions.parquet")
    healthy = [c for c in rows if c.condition == HEALTHY]

    vecs = embed_prompts(queries)
    calib_idx = [i for i, q in enumerate(queries) if q.split == "calib"]
    centres = kmeans(vecs[calib_idx], K, s.corpus_seed)
    cluster = np.argmin(((vecs[:, None, :] - centres[None]) ** 2).sum(2), axis=1)
    of = {q.query_id: int(cluster[i]) for i, q in enumerate(queries)}

    lengths: dict[int, list[int]] = {j: [] for j in range(K)}
    split_of = {q.query_id: q.split for q in queries}
    for c in healthy:
        if c.sample_idx == 0 and split_of[c.query_id] == "calib":
            lengths[of[c.query_id]].append(len((c.text or "").split()))
    order = sorted(range(K), key=lambda j: np.median(lengths[j]), reverse=True)
    pool: dict[int, list] = {j: [] for j in range(K)}
    for q in queries:
        if q.split in ("calib", "fault"):
            pool[of[q.query_id]].append(q)
    shifts = {
        "F4": draw(pool, order[:3], s.corpus_seed, 4),
        "F6": draw(pool, order[3:], s.corpus_seed, 6),
    }
    derived = {
        name: [
            c.model_copy(update={"condition": name})
            for c in healthy
            if c.query_id in {q.query_id for q in qs}
        ]
        for name, qs in shifts.items()
    }
    clusters = {
        j: {"size": int((cluster == j).sum()), "median_answer_words": float(np.median(lengths[j]))}
        for j in order
    }

    out: dict = {"clusters": clusters, "long_clusters": order[:3], "monitors": {}}
    runs = replay_runs(
        list(derived), 200, DEFAULT_REPLAY_WINDOWS, DEFAULT_MIN_PREFIX, s.calibration_seed
    )

    calib_q, calib_rows = calibration_rows(queries, rows)
    record = freeze.verify(ROOT / "thresholds.json", calib_rows)
    names = list(record.thresholds)
    emb = read_embeddings(ROOT / "embeddings.parquet")
    for name, d in derived.items():
        emb = alias_embeddings(emb, d, name)
    signals = fit_signals(
        build_monitor_signals(emb, names, record.scoring),
        calib_q,
        calib_rows,
    )
    fault_q = sorted((q for q in queries if q.split == "fault"), key=lambda q: q.query_id)
    recs = run_replay(runs, healthy, derived, fault_q, signals, record.thresholds)
    out["monitors"]["frozen pooled"] = _summary(recs)

    pairs = _paired_inputs("chat", "rounds")
    ref_s = paired.SAMPLES["rounds"][1]
    reference_rows = read_completions(ROOT / "reference" / "completions.parquet")
    ref_any = {
        **paired._groups([c for c in reference_rows if c.condition == "R"], ref_s),
        **paired._groups([c for c in rows if c.condition == "F7"], ref_s),
    }
    vectors = paired._vectors(emb)
    vectors.update(paired._vectors(read_embeddings(ROOT / "reference" / "embeddings.parquet")))
    pool_table = next(iter(pairs.tables.values()))[0]
    for name, d in derived.items():
        qids = sorted({c.query_id for c in d})
        pairs.faulty[name] = d
        pairs.tables[name] = (
            pool_table,
            paired.build_table(paired._groups(d, ref_s), ref_any, vectors, qids),
        )
    rounds = json.loads((DATA_DIR / "paired" / "chat_rounds.json").read_text())["thresholds"]
    recs = paired.replay(runs, pairs.pool, pairs.healthy, pairs.faulty, pairs.tables, rounds)
    out["monitors"]["paired rounds"] = _summary(recs)

    # Added 22 Oct 2026: the served-answer monitor (config/served_study.yaml),
    # primary kernel, on the same streams. Nothing above changes.
    from evaluate.served import calibrate_served, replay_served, served_tables

    st = served_tables(pairs, rows, reference_rows, vectors, "served_hamming")
    th = calibrate_served(
        st.calib,
        pairs.calib,
        s.false_alarm_budget,
        s.bootstrap_windows,
        s.calibration_seed,
        "served_hamming",
    )["thresholds"]["served_hamming"]
    recs = replay_served(runs, pairs, st.conditions, "served_hamming", th)
    out["monitors"]["served answer"] = _summary(recs)

    import met_baseline as met  # rebinds paired.build_table, window_scores, STATISTICS

    mpairs = _paired_inputs("chat", "rounds")
    pool_table = next(iter(mpairs.tables.values()))[0]
    for name, d in derived.items():
        qids = sorted({c.query_id for c in d})
        mpairs.faulty[name] = d
        mpairs.tables[name] = (
            pool_table,
            paired.build_table(paired._groups(d, ref_s), ref_any, vectors, qids),
        )
    mrec = paired.calibrate_over_rounds(
        [mpairs.calib_table, *mpairs.round_tables],
        mpairs.calib,
        s.false_alarm_budget,
        s.bootstrap_windows,
        s.calibration_seed,
        paired.STATISTICS,
    )
    recs = paired.replay(
        runs, mpairs.pool, mpairs.healthy, mpairs.faulty, mpairs.tables, mrec["thresholds"]
    )
    out["monitors"]["MET baseline"] = _summary(recs)
    assert met

    (ROOT / "results" / "topics.json").write_text(json.dumps(out, indent=2) + "\n")
    lines = [
        "# Real traffic: topic shifts found without labels\n",
        f"K = {K} prompt clusters; long-answer clusters {order[:3]}. Each shift: {SIZE} questions, "
        f"{SHARE:.0%} from one half. Share of 200 streams alarmed after the onset (Wilson 95%).\n",
        "| monitor | F4 long-topic shift | F6 short-topic shift | pre-onset false alarms |",
        "|---|---|---|---|",
    ]
    for m, r in out["monitors"].items():
        lines.append(f"| {m} | {r['F4']} | {r['F6']} | {r['pre_onset']} |")
    md = "\n".join(lines) + "\n"
    (DATA_DIR / "report_paired" / "chat_topics.md").write_text(md)
    print(json.dumps(clusters, indent=1))  # noqa: T201
    print(md)  # noqa: T201


def _summary(records: list[dict]) -> dict:
    out = {}
    for cond in ("F4", "F6"):
        rs = [r for r in records if r["condition"] == cond]
        k = sum(r["latency"] is not None for r in rs)
        lo, hi = wilson_interval(k, len(rs))
        out[cond] = f"{100 * k / len(rs):.1f}% ({100 * lo:.1f}-{100 * hi:.1f})"
    pw = sum(r["pre_onset_windows"] for r in records)
    pa = sum(r["pre_onset_alarms"] for r in records)
    out["pre_onset"] = f"{100 * pa / pw:.3f}%"
    return out


if __name__ == "__main__":
    main()

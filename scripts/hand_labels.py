"""Usage: uv run python scripts/hand_labels.py build | score.

The hand-labelled check of the automatic accuracy scoring
(config/hand_label_study.yaml).

build  draws the 160 items, writes one blind sheet per labeller to
       data/hand_labels/sheet_<name>.csv and the key (condition, model, checker
       verdict) to data/hand_labels/key.json, and prints the key's sha256.
       Refuses to overwrite an existing key.
score  reads the filled sheets and the key and writes
       data/hand_labels/results.json and results.md.
"""

import csv
import hashlib
import json
import random
import sys

import numpy as np

from corpus.build import read_queries
from evaluate.metrics import wilson_interval
from oracle.score import score_completion
from provider.collect import read_completions
from settings import DATA_DIR, load_api_study

SPEC = load_api_study("hand_label_study")
OUT = DATA_DIR / "hand_labels"
KEY = OUT / "key.json"
FIELDS = ["item", "slice", "question", "reference_answer", "model_answer", "label", "note"]


def _question(prompt: str, slice_name: str) -> str:
    # banking77's 77-label list is the same on every item; the gold label is shown
    if slice_name == "banking77":
        return "\n".join(ln for ln in prompt.splitlines() if not ln.startswith("Labels:"))
    return prompt


def build() -> None:
    """Draw the sample, write the blind sheets and the key."""
    if KEY.exists():
        raise SystemExit(f"{KEY} exists; the sample is drawn once")
    rng = random.Random(SPEC["seed"])
    queries = {q.query_id: q for q in read_queries(DATA_DIR / "queries.jsonl")}
    rows = [
        c
        for c in read_completions(DATA_DIR / "completions.parquet")
        if c.condition in SPEC["conditions"]
        and c.sample_idx == SPEC["sample_idx"]
        and queries[c.query_id].split == SPEC["split"]
        and queries[c.query_id].slice in SPEC["slices"]
    ]
    items = []
    for s in SPEC["slices"]:
        for cond in SPEC["conditions"]:
            pool = sorted(
                (c for c in rows if c.condition == cond and queries[c.query_id].slice == s),
                key=lambda c: c.query_id,
            )
            items += rng.sample(pool, SPEC["per_slice_condition"])
    rng.shuffle(items)

    names, own, extra = SPEC["labellers"], SPEC["own_items"], SPEC["overlap_items"]
    assert len(items) == own * len(names)
    key, sheets = {}, {n: [] for n in names}
    for i, c in enumerate(items):
        item = f"HL{i:03d}"
        q = queries[c.query_id]
        primary = names[i // own]
        key[item] = {
            "query_id": c.query_id,
            "slice": q.slice,
            "condition": c.condition,
            "model_id": c.model_id,
            "checker": int(bool(score_completion(c, q))),
            "primary": primary,
        }
        row = {
            "item": item,
            "slice": q.slice,
            "question": _question(q.prompt, q.slice),
            "reference_answer": q.ground_truth,
            "model_answer": c.text,
            "label": "",
            "note": "",
        }
        sheets[primary].append(row)
        # the first `extra` items of each labeller go to the next one in the ring
        if i % own < extra:
            second = names[(i // own + 1) % len(names)]
            key[item]["second"] = second
            sheets[second].append(row)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, sheet in sheets.items():
        random.Random(f"{SPEC['seed']}:{name}").shuffle(sheet)
        with (OUT / f"sheet_{name}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(sheet)
    KEY.write_text(json.dumps(key, indent=1, sort_keys=True) + "\n")
    print(f"{len(items)} items, {sum(len(s) for s in sheets.values())} labels")  # noqa: T201
    print(f"key sha256 {hashlib.sha256(KEY.read_bytes()).hexdigest()}")  # noqa: T201


def _kappa(a: list[int], b: list[int]) -> float:
    a, b = np.array(a), np.array(b)
    po = float((a == b).mean())
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson_interval(k, n)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"), "ci_lo": lo, "ci_hi": hi}


def score() -> None:
    """Compare the filled sheets with the checker and write the results."""
    key = json.loads(KEY.read_text())
    labels: dict[str, dict[str, str]] = {}
    for name in SPEC["labellers"]:
        with (OUT / f"sheet_{name}.csv").open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                labels.setdefault(row["item"], {})[name] = row["label"].strip()

    pairs, unclear, missing, human_pairs = [], 0, 0, []
    for item, k in key.items():
        got = labels.get(item, {})
        first = got.get(k["primary"], "")
        if first == "":
            missing += 1
            continue
        if first == "?":
            unclear += 1
            continue
        pairs.append((k["slice"], k["condition"], k["checker"], int(first)))
        second = got.get(k.get("second", ""), "")
        if second in ("0", "1"):
            human_pairs.append((int(first), int(second)))
    if missing:
        print(f"warning: {missing} items have no primary label yet")  # noqa: T201

    def agree(sel: list[tuple]) -> dict:
        return _rate(sum(c == h for *_, c, h in sel), len(sel))

    out = {
        "items": len(key),
        "scored": len(pairs),
        "unclear": unclear,
        "missing": missing,
        "agreement": agree(pairs),
        "kappa": _kappa([p[2] for p in pairs], [p[3] for p in pairs]),
        "false_pass": _rate(sum(c == 1 and h == 0 for *_, c, h in pairs), len(pairs)),
        "false_fail": _rate(sum(c == 0 and h == 1 for *_, c, h in pairs), len(pairs)),
        "by_slice": {s: agree([p for p in pairs if p[0] == s]) for s in SPEC["slices"]},
        "by_condition": {c: agree([p for p in pairs if p[1] == c]) for c in SPEC["conditions"]},
        "human_human": {
            "n": len(human_pairs),
            "agreement": _rate(sum(a == b for a, b in human_pairs), len(human_pairs)),
            "kappa": _kappa([a for a, _ in human_pairs], [b for _, b in human_pairs]),
        },
    }

    # healthy-minus-F1 accuracy gap by checker and by person, paired bootstrap
    hi, lo = SPEC["conditions"]
    arr = {c: np.array([(p[2], p[3]) for p in pairs if p[1] == c]) for c in (hi, lo)}

    def gaps(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
        return a[:, 0].mean() - b[:, 0].mean(), a[:, 1].mean() - b[:, 1].mean()

    g_checker, g_human = gaps(arr[hi], arr[lo])
    rng = np.random.default_rng(SPEC["seed"])
    diffs = []
    for _ in range(SPEC["bootstrap_resamples"]):
        a = arr[hi][rng.integers(0, len(arr[hi]), len(arr[hi]))]
        b = arr[lo][rng.integers(0, len(arr[lo]), len(arr[lo]))]
        gc, gh = gaps(a, b)
        diffs.append(gc - gh)
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    out["gap"] = {
        "checker": g_checker,
        "human": g_human,
        "difference": g_checker - g_human,
        "ci_lo": float(d_lo),
        "ci_hi": float(d_hi),
    }
    out["criteria"] = {
        "HL1": out["agreement"]["rate"] >= SPEC["criteria"]["agreement_min"],
        "HL2": bool(d_lo <= 0 <= d_hi),
    }
    (OUT / "results.json").write_text(json.dumps(out, indent=1) + "\n")

    def pct(r: dict) -> str:
        return f"{100 * r['rate']:.1f}% ({r['k']}/{r['n']}, 95% {100 * r['ci_lo']:.1f}-{100 * r['ci_hi']:.1f})"

    md = [
        "# Hand-labelled check of the automatic scoring\n",
        f"{out['scored']} items scored, {unclear} unclear, {missing} missing.\n",
        f"- Checker-person agreement: {pct(out['agreement'])}; kappa {out['kappa']:.3f}",
        f"- Checker false passes: {pct(out['false_pass'])}; false fails: {pct(out['false_fail'])}",
        f"- Person-person agreement on the overlap: {pct(out['human_human']['agreement'])}; "
        f"kappa {out['human_human']['kappa']:.3f}",
        f"- Healthy-minus-F1 gap: checker {100 * g_checker:+.1f}, person {100 * g_human:+.1f} points; "
        f"difference {100 * (g_checker - g_human):+.1f} (95% {100 * d_lo:+.1f} to {100 * d_hi:+.1f})",
        f"- HL1 (agreement >= 90%): {'met' if out['criteria']['HL1'] else 'NOT met'}; "
        f"HL2 (no gap bias): {'met' if out['criteria']['HL2'] else 'NOT met'}\n",
        "| slice | agreement |",
        "|---|---|",
        *[f"| {s} | {pct(r)} |" for s, r in out["by_slice"].items()],
        *[f"| {c} (all slices) | {pct(r)} |" for c, r in out["by_condition"].items()],
    ]
    (OUT / "results.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))  # noqa: T201


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()

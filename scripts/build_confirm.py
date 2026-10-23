"""Usage: uv run python scripts/build_confirm.py

Builds data/confirm for the blind confirmatory study (config/confirm_study.yaml):
the local study's questions, frozen records, healthy and F7 answers and their
embeddings, plus C1-C3, which are the H3-H5 answers to the fault questions
relabelled (their model ids and digests are kept: they are those models'
answers). C4 and C5 are generated into it afterwards with `generate`. Refuses to
overwrite an existing data/confirm.
"""

import shutil

import pandas as pd

from corpus.build import read_queries
from corpus.splits import fault_subset
from settings import DATA_DIR, load_api_study

SRC, OUT = DATA_DIR, DATA_DIR / "confirm"
SOURCES = {
    c: spec["source"]
    for c, spec in load_api_study("confirm_study")["confirm"]["conditions"].items()
    if "source" in spec
}

if OUT.exists():
    raise SystemExit(f"{OUT} exists; it is built once")
OUT.mkdir()
for name in ("queries.jsonl", "thresholds.json", "thresholds_v3.json"):
    shutil.copy2(SRC / name, OUT / name)

fault = {q.query_id for q in fault_subset(read_queries(SRC / "queries.jsonl"))}
rows = pd.read_parquet(SRC / "completions.parquet")
parts = [rows[rows.condition.isin(["healthy", "F7"])]]
for cond, src in SOURCES.items():
    part = rows[(rows.condition == src) & rows.query_id.isin(fault)].copy()
    part["condition"] = cond
    parts.append(part)
pd.concat(parts, ignore_index=True).to_parquet(OUT / "completions.parquet", index=False)

emb = pd.read_parquet(SRC / "embeddings.parquet")
key = emb.custom_id.str.split("|")
parts = [emb[key.str[2].isin(["healthy", "F7"])]]
for cond, src in SOURCES.items():
    part = emb[(key.str[2] == src) & key.str[0].isin(fault)].copy()
    part["custom_id"] = part.custom_id.str.replace(f"|{src}|", f"|{cond}|", regex=False)
    parts.append(part)
pd.concat(parts, ignore_index=True).to_parquet(OUT / "embeddings.parquet", index=False)

done = pd.read_parquet(OUT / "completions.parquet").condition.value_counts().to_dict()
print(f"{OUT}: {done}")

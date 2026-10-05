"""Corpus construction. Track T1.

Assembles the 5,000-query stream from six HuggingFace datasets plus one locally
synthesised slice. Composition and per-split counts are fixed in
`settings.SLICE_SIZES` / `settings.SPLIT_SIZES` - read them from there, never
restate the numbers here.

`datasets` is used for corpus construction ONLY. It must never pull a model
(C7); if a loader tries, that is a bug, not a convenience.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from corpus.schemas import Query
from settings import SLICE_SIZES, SLICES, VERIFIABLE_SLICES, slice_template

#: HuggingFace dataset coordinates per slice. `jsonx` is absent by design - it
#: has no upstream dataset and is synthesised by `build_jsonx_slice`.
#:
#: `eli5` is DEPRECATED on the Hub and its canonical config no longer loads.
#: Resolve it at implementation time and RECORD the decision here in a comment
#: - an undocumented silent substitution makes the xsum/eli5 half of the corpus
#: unreproducible, and those two slices are 44% of every window.
#:
#: Decisions recorded per slice:
#:
#: - eli5: the original `eli5` dataset is gone from the Hub. We use
#:   `sentence-transformers/eli5` (325k question/answer pairs scraped from the
#:   same subreddit). Only the QUESTION is used, so the substitution affects
#:   prompts, not scoring - eli5 is not a verifiable slice.
#: - banking77: `PolyAI/banking77` is script-based and no longer loads under
#:   datasets>=3. `legacy-datasets/banking77` is the Hub's own parquet mirror of
#:   the same 3,080-row test set, with the 77 intent names as a ClassLabel.
#: - triviaqa: `rc.nocontext` - closed-book. No evidence documents are given,
#:   which keeps answer variance high and leaves S3 headroom.
#: - squad2: unanswerable questions are IN SCOPE at their natural rate (~50% of
#:   the validation split). Their ground truth is the literal string
#:   "unanswerable", which is what the slice template tells the model to reply.
#:   That reply does not match any refusal pattern, so it does not inflate S2's
#:   healthy baseline.
HF_SOURCES: dict[str, tuple[str, str | None, str]] = {
    "gsm8k": ("openai/gsm8k", "main", "test"),
    "triviaqa": ("mandarjoshi/trivia_qa", "rc.nocontext", "validation"),
    "squad2": ("rajpurkar/squad_v2", None, "validation"),
    "banking77": ("legacy-datasets/banking77", None, "test"),
    "xsum": ("EdinburghNLP/xsum", None, "test"),
    "eli5": ("sentence-transformers/eli5", None, "train"),
}

#: Exact Hub commit per source. Without a pin, an upstream edit silently changes
#: which records a seed selects, and the corpus stops being reproducible.
HF_REVISIONS: dict[str, str] = {
    "gsm8k": "740312add88f781978c0658806c59bc2815b9866",
    "triviaqa": "0f7faf33a3908546c6fd5b73a660e0f8ff173c2f",
    "squad2": "3ffb306f725f7d2ce8394bc1873b24868140c412",
    "banking77": "f54121560de48f2852f90be299010d1d6dc612ec",
    "xsum": "7d4d486c2f8ef850b1a11aead99b894ff3dd7da9",
    "eli5": "6e3b20a7a427560845779f217c3ff71493374bc0",
}

#: XSum documents above this many words are not "usable". The test split runs to
#: 10,315 words; an unbounded document makes prompt length a confound and lets a
#: handful of rows dominate prompt-processing time. 800 words (~1,050 tokens)
#: keeps ~90% of the split.
XSUM_MAX_WORDS: int = 800

#: Placeholder until `splits.assign_splits` runs.
UNASSIGNED_SPLIT: str = "unassigned"

_GSM8K_ANSWER = re.compile(r"####\s*(.+?)\s*$")

#: Signature of UTF-8 text that was decoded as cp1252 somewhere upstream. ~11%
#: of XSum carries it ("Â£30bn" for "£30bn").
_MOJIBAKE = re.compile("Ã|Â|â€")


def load_slice(slice_name: str, n: int, seed: int) -> list[Query]:
    """Load and normalise `n` queries for one slice.

    Behaviour:
        Pull the slice's source via `datasets.load_dataset`, take a deterministic
        sample of size `n` under `seed`, and map each record to a `Query`:
        build the prompt from `settings.slice_template(slice_name)` - never
        inline prompt text here - extract `ground_truth` for verifiable slices, and
        attach `label_space` for banking77. `split` is left as a placeholder -
        `splits.assign_splits` fills it.

    Args:
        slice_name: one of settings.SLICES.
        n: exactly settings.SLICE_SIZES[slice_name].
        seed: sampling seed; the same seed must always yield the same queries.

    Returns:
        Exactly `n` Query objects with unique `query_id`s of the form
        '<slice>_<5-digit index>'.

    Raises:
        ValueError: if the source yields fewer than `n` usable records. Do NOT
            pad or resample to reach `n` - a short slice breaks the window
            strata and must surface loudly.

    Constraints:
        The Query validator enforces the ground_truth / label_space invariants,
        so a mis-mapped field fails here rather than three stages later.

    Edge cases:
        - squad2 contains deliberately unanswerable questions. Decide whether
          they are in scope and document it: they change the refusal baseline
          for S2, which is calibrated on this corpus.
        - triviaqa answers have aliases. Pick one canonical form for
          `ground_truth` and let the ORACLE handle alias matching - the corpus
          must not encode scoring policy.

    """
    if slice_name not in HF_SOURCES:
        raise ValueError(f"No HuggingFace source for slice {slice_name!r}")
    dataset = _load_source(slice_name)
    order = np.random.default_rng(_slice_seed(seed, slice_name)).permutation(len(dataset))
    template = slice_template(slice_name)
    label_names = _banking77_labels() if slice_name == "banking77" else None

    queries: list[Query] = []
    seen_prompts: set[str] = set()
    for idx in order:
        fields = _normalise_record(slice_name, dataset[int(idx)], label_names)
        if fields is None:
            continue
        prompt = template.format(**fields["template_fields"]).strip()
        if prompt in seen_prompts:
            continue
        seen_prompts.add(prompt)
        queries.append(
            Query(
                query_id=f"{slice_name}_{len(queries):05d}",
                slice=slice_name,
                prompt=prompt,
                ground_truth=fields["ground_truth"],
                label_space=label_names,
                split=UNASSIGNED_SPLIT,
            )
        )
        if len(queries) == n:
            return queries
    raise ValueError(
        f"slice {slice_name!r}: source yielded only {len(queries)} usable records, need {n}"
    )


def build_jsonx_slice(n: int, seed: int) -> list[Query]:
    """Synthesise the `jsonx` slice locally. No upstream dataset exists.

    Behaviour:
        Generate `n` prompts asking for a JSON object conforming to
        `config/jsonx_schema.json`, with varied but deterministic content. The
        `ground_truth` is the canonical JSON object the prompt describes.

    Constraints:
        These rows are submitted with NO `response_format`. Strict json_schema
        mode makes the PLATFORM validate and fail the row server-side, so
        malformed output would never come back as text and S5 would have
        nothing left to measure. The prompt must therefore state the schema in
        prose, and the schema must be reachable by a healthy model but not
        trivially so - if healthy validity is 100%, S5 has no headroom to
        detect degradation.
    """
    rng = np.random.default_rng(_slice_seed(seed, "jsonx"))
    template = slice_template("jsonx")
    queries: list[Query] = []
    seen: set[str] = set()
    attempts = 0
    while len(queries) < n:
        attempts += 1
        if attempts > n * 50:
            raise ValueError(f"jsonx: could not synthesise {n} distinct items")
        description, truth = _synth_jsonx_item(rng)
        if description in seen:
            continue
        seen.add(description)
        queries.append(
            Query(
                query_id=f"jsonx_{len(queries):05d}",
                slice="jsonx",
                prompt=template.format(question=description).strip(),
                ground_truth=json.dumps(truth, sort_keys=True),
                label_space=None,
                split=UNASSIGNED_SPLIT,
            )
        )
    return queries


def build_corpus(seed: int) -> list[Query]:
    """Assemble all seven slices into the full 5,000-query corpus.

    Returns:
        Exactly 5,000 Query objects, ordered deterministically by
        (slice, query_id), with `split` unassigned.

    Raises:
        ValueError: if the total is not 5,000, or the verifiable slices do not
            sum to exactly 2,800. Assert both - they are the Stage 1 gate.

    """
    return list(_build_corpus_cached(seed))


def write_queries(queries: list[Query], path: Path) -> int:
    """Write `queries.jsonl`, one JSON object per line. Returns rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for q in queries:
            fh.write(q.model_dump_json() + "\n")
    return len(queries)


def read_queries(path: Path) -> list[Query]:
    """Read and re-validate `queries.jsonl`. Raises on any invalid row."""
    queries: list[Query] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                queries.append(Query.model_validate_json(line))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: invalid query row: {exc}") from exc
    return queries


# ── Private helpers ─────────────────────────────────────────────────────────


@functools.lru_cache(maxsize=4)
def _build_corpus_cached(seed: int) -> tuple[Query, ...]:
    """Build once per seed per process. Query is frozen, so sharing is safe."""
    queries: list[Query] = []
    for slice_name in SLICES:
        n = SLICE_SIZES[slice_name]
        if slice_name == "jsonx":
            queries.extend(build_jsonx_slice(n, seed))
        else:
            queries.extend(load_slice(slice_name, n, seed))
    queries.sort(key=lambda q: (q.slice, q.query_id))

    total = len(queries)
    verifiable = sum(1 for q in queries if q.slice in VERIFIABLE_SLICES)
    expected_total = sum(SLICE_SIZES.values())
    expected_verifiable = sum(SLICE_SIZES[s] for s in VERIFIABLE_SLICES)
    if total != expected_total:
        raise ValueError(f"corpus has {total} queries, expected {expected_total}")
    if verifiable != expected_verifiable:
        raise ValueError(
            f"corpus has {verifiable} verifiable queries, expected {expected_verifiable}"
        )
    return tuple(queries)


def _slice_seed(seed: int, slice_name: str) -> np.random.SeedSequence:
    """Independent, stable stream per slice: adding a slice never reshuffles another."""
    return np.random.SeedSequence([seed, SLICES.index(slice_name)])


@functools.cache
def _load_source(slice_name: str) -> Any:
    """Load one pinned HuggingFace split. Cached per process; HF caches on disk."""
    import datasets

    path, config, split = HF_SOURCES[slice_name]
    return datasets.load_dataset(path, config, split=split, revision=HF_REVISIONS[slice_name])


@functools.lru_cache(maxsize=1)
def _banking77_labels() -> list[str]:
    """The 77 intent names, in ClassLabel order - checked against the config file.

    S6 reads the label space from config/banking77_labels.yaml because a
    Completion does not carry it. If the dataset and that file ever disagree,
    S6 would map completions against a different label set than the prompts
    offered, so the build fails rather than proceeding.
    """
    from textnorm import load_banking77_labels

    names = list(_load_source("banking77").features["label"].names)
    if names != load_banking77_labels():
        raise ValueError(
            "banking77 labels in the pinned dataset differ from config/banking77_labels.yaml"
        )
    return names


def _normalise_record(
    slice_name: str, record: dict[str, Any], label_names: list[str] | None
) -> dict[str, Any] | None:
    """Map one raw record to template fields + ground truth, or None if unusable."""
    if slice_name == "gsm8k":
        match = _GSM8K_ANSWER.search(record["answer"].strip())
        question = record["question"].strip()
        if not match or not question:
            return None
        answer = match.group(1).replace(",", "").strip()
        return {"template_fields": {"question": question}, "ground_truth": answer}

    if slice_name == "triviaqa":
        question = record["question"].strip()
        answer = (record["answer"] or {}).get("value", "").strip()
        if not question or not answer:
            return None
        return {"template_fields": {"question": question}, "ground_truth": answer}

    if slice_name == "squad2":
        question = record["question"].strip()
        context = record["context"].strip()
        if not question or not context:
            return None
        texts = [t.strip() for t in record["answers"]["text"] if t.strip()]
        answer = texts[0] if texts else "unanswerable"
        return {
            "template_fields": {"question": question, "context": context},
            "ground_truth": answer,
        }

    if slice_name == "banking77":
        text = record["text"].strip()
        if not text or label_names is None:
            return None
        return {
            "template_fields": {"question": text, "label_space": ", ".join(label_names)},
            "ground_truth": label_names[int(record["label"])],
        }

    if slice_name == "xsum":
        document = _repair_mojibake(record["document"].strip())
        words = len(document.split())
        if words == 0 or words > XSUM_MAX_WORDS:
            return None
        return {"template_fields": {"context": document}, "ground_truth": None}

    if slice_name == "eli5":
        question = record["question"].strip()
        if not question:
            return None
        return {"template_fields": {"question": question}, "ground_truth": None}

    raise ValueError(f"No normaliser for slice {slice_name!r}")


def _repair_mojibake(text: str) -> str:
    """Undo a cp1252 mis-decode when present; leave the text alone if it won't round-trip."""
    if not _MOJIBAKE.search(text):
        return text
    try:
        return text.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


# ── jsonx synthesis ─────────────────────────────────────────────────────────
#
# Each item is described in PROSE that has to be interpreted, not copied, so a
# healthy model does not hit 100% validity and S5 keeps headroom:
#   - the category is implied by the subject matter, never named as an enum value;
#   - the priority is given in words and must be mapped to an integer 1-5;
#   - the description offers up to seven keywords, but `tags` allows at most five;
#   - publication status is phrased indirectly ("still a draft", "went live").

_JSONX_TOPICS: dict[str, list[tuple[str, list[str]]]] = {
    "finance": [
        ("Budgeting for your first apartment", ["rent", "savings", "budget", "deposit"]),
        ("Index funds versus active managers", ["investing", "fees", "markets", "funds"]),
        ("Paying down credit card debt", ["debt", "interest", "credit", "payments"]),
        ("Understanding your payslip deductions", ["tax", "salary", "pension", "payroll"]),
        ("Building an emergency fund", ["savings", "cash", "risk", "planning"]),
        ("How mortgage rates are set", ["mortgage", "rates", "banks", "housing"]),
    ],
    "health": [
        ("Getting better sleep after night shifts", ["sleep", "shift work", "rest", "routine"]),
        ("Beginner strength training at home", ["exercise", "strength", "fitness", "home"]),
        ("Reading nutrition labels", ["nutrition", "food", "labels", "diet"]),
        ("Managing seasonal allergies", ["allergies", "pollen", "symptoms", "spring"]),
        ("Hydration during long runs", ["running", "hydration", "endurance", "water"]),
        ("Desk posture and back pain", ["posture", "back pain", "office", "ergonomics"]),
    ],
    "technology": [
        ("Setting up a home mesh network", ["wifi", "networking", "router", "home"]),
        ("Choosing a password manager", ["security", "passwords", "privacy", "apps"]),
        ("What a GPU actually does", ["hardware", "gpu", "graphics", "computing"]),
        ("Backing up your phone photos", ["backup", "photos", "cloud", "storage"]),
        ("An introduction to version control", ["git", "software", "coding", "teams"]),
        ("Why your laptop battery degrades", ["battery", "laptops", "hardware", "charging"]),
    ],
    "travel": [
        ("Packing light for a two-week trip", ["packing", "luggage", "travel", "tips"]),
        ("Rail passes across Europe", ["trains", "europe", "rail", "itinerary"]),
        ("Avoiding jet lag on long flights", ["flights", "jet lag", "sleep", "timezones"]),
        ("Budget hostels in South East Asia", ["hostels", "asia", "backpacking", "budget"]),
        ("Renting a car abroad", ["car hire", "driving", "insurance", "abroad"]),
        ("Visiting national parks off-season", ["parks", "hiking", "nature", "seasons"]),
    ],
    "education": [
        ("Spaced repetition for language learning", ["languages", "memory", "study", "flashcards"]),
        (
            "Writing a strong college application essay",
            ["essays", "admissions", "writing", "college"],
        ),
        (
            "Teaching fractions with everyday objects",
            ["maths", "teaching", "fractions", "classroom"],
        ),
        ("Taking effective lecture notes", ["notes", "lectures", "study", "university"]),
        ("Choosing an online course platform", ["online learning", "courses", "mooc", "skills"]),
        ("Preparing for open-book exams", ["exams", "revision", "study", "assessment"]),
    ],
}

_JSONX_FORMATS: tuple[str, ...] = (
    "article",
    "guide",
    "newsletter issue",
    "explainer",
    "short video script",
)

_JSONX_PRIORITY_WORDS: dict[int, tuple[str, ...]] = {
    1: ("lowest priority", "whenever there is time, no rush at all"),
    2: ("low priority", "nice to have but not pressing"),
    3: ("medium priority", "a normal place in the queue"),
    4: ("high priority", "should go out soon"),
    5: ("top priority", "urgent, needed immediately"),
}

_JSONX_EXTRA_KEYWORDS: tuple[str, ...] = (
    "beginners",
    "checklist",
    "explained",
    "2026",
    "faq",
    "practical",
    "overview",
)

_JSONX_PUBLISHED_PHRASES: dict[bool, tuple[str, ...]] = {
    True: ("it went live last week", "it is already available to readers", "it has been released"),
    False: ("it is still a draft", "it has not been released yet", "it is waiting for review"),
}


def _synth_jsonx_item(rng: np.random.Generator) -> tuple[str, dict[str, Any]]:
    """One prose item description and the canonical JSON object it describes."""
    categories = sorted(_JSONX_TOPICS)
    category = categories[int(rng.integers(len(categories)))]
    topics = _JSONX_TOPICS[category]
    title, base_tags = topics[int(rng.integers(len(topics)))]
    fmt = _JSONX_FORMATS[int(rng.integers(len(_JSONX_FORMATS)))]
    priority = int(rng.integers(1, 6))
    priority_phrase = _JSONX_PRIORITY_WORDS[priority][int(rng.integers(2))]
    published = bool(rng.integers(2))
    published_phrase = _JSONX_PUBLISHED_PHRASES[published][int(rng.integers(3))]

    n_extra = int(rng.integers(0, 4))
    extras = [str(x) for x in rng.choice(_JSONX_EXTRA_KEYWORDS, size=n_extra, replace=False)]
    keywords = list(base_tags) + extras

    description = (
        f'A {fmt} called "{title}". Relevant keywords: {", ".join(keywords)}. '
        f"It is {priority_phrase}, and {published_phrase}."
    )
    truth = {
        "title": title,
        "category": category,
        "priority": priority,
        "tags": keywords[:5],
        "published": published,
    }
    return description, truth

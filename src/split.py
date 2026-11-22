"""Stage `split` (Member B): leak-free train/val/test split of the validated Bitext data.

Reads  data/interim/validated.parquet, configs/labels.yaml, configs/schema.yaml
Writes data/processed/{train,val,test}.parquet and reports/split_report.json

Why this is not a random row split
----------------------------------
Bitext queries are generated from templates, so the same sentence appears many times with one word
swapped or a typo ("i want assistance canceling order" / "i need assistance canceling order").
A random split puts these twins in train AND test, so the model is graded on questions it has
effectively already seen. So we:

1. normalise each query (lowercase, drop {{placeholders}}, digits, punctuation, extra spaces);
2. inside each intent, link two queries if their TF-IDF character n-gram cosine similarity is
   >= split.group_threshold (character n-grams still match through typos);
3. take connected components of that graph as `group_id` - every pair of twins ends up in the
   same group, so no pair above the threshold can ever cross splits (tests/test_split.py checks it);
4. assign whole groups to train/val/test with a greedy per-intent fill: biggest groups first, each
   into the split furthest below its target share. This keeps every intent (and therefore the
   escalation positive rate, which is a function of intent) close to 70/15/15.

Grouping is done within an intent because a twin with a *different* intent does not leak the
answer, and it keeps the similarity matrices small (~1000x1000 per intent).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

from src.labels import add_labels, check_against_schema, load_label_config

SPLITS = ("train", "val", "test")
OUT_COLUMNS = ["row_id", "query", "intent", "category", "group_id", "label_escalate", "label_rule"]

_PLACEHOLDER = re.compile(r"\{\{[^}]*\}\}")
_DIGITS = re.compile(r"\d+")
_PUNCT = re.compile(r"[^\w\s]")
_SPACES = re.compile(r"\s+")


def normalise(text: str) -> str:
    s = text.lower()
    s = _PLACEHOLDER.sub(" ", s)
    s = _DIGITS.sub(" ", s)
    s = _PUNCT.sub(" ", s)
    return _SPACES.sub(" ", s).strip()


def assign_groups(df: pd.DataFrame, threshold: float, ngram_range=(3, 5)) -> np.ndarray:
    """group_id per row: connected components of the within-intent similarity >= threshold graph."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(ngram_range), sublinear_tf=True)
    X = vec.fit_transform(df["query_norm"])          # rows are L2-normalised, so X @ X.T = cosine
    group_id = np.empty(len(df), dtype=np.int64)
    offset = 0
    for intent in sorted(df["intent"].unique()):     # sorted: same ids on every run
        idx = np.flatnonzero(df["intent"].to_numpy() == intent)
        sim = X[idx] @ X[idx].T
        adj = csr_matrix(sim >= threshold)
        n, labels = connected_components(adj, directed=False)
        group_id[idx] = labels + offset
        offset += n
    return group_id


def assign_splits(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> pd.Series:
    """Greedy per-intent fill of whole groups into train/val/test. Deterministic for a given seed."""
    frac = np.array([fractions[s] for s in SPLITS], dtype=float)
    rng = np.random.default_rng(seed)
    group_split: dict[int, str] = {}
    for intent in sorted(df["intent"].unique()):
        sizes = df.loc[df["intent"] == intent, "group_id"].value_counts().sort_index()
        keys = sizes.index.to_numpy().copy()
        rng.shuffle(keys)                              # random order among equal-sized groups
        keys = sorted(keys, key=lambda k: -sizes[k])   # stable sort: biggest groups first
        target = frac * sizes.sum()
        filled = np.zeros(len(SPLITS))
        for k in keys:
            s = int(np.argmax(target - filled))        # split with the largest remaining deficit
            group_split[int(k)] = SPLITS[s]
            filled[s] += sizes[k]
    return df["group_id"].map(group_split)


def build_splits(validated: pd.DataFrame, label_cfg: dict, cfg: dict, seed: int) -> pd.DataFrame:
    df = add_labels(validated, label_cfg)
    df["query"] = df["instruction"]
    df["query_norm"] = df["query"].map(normalise)
    df["group_id"] = assign_groups(df, cfg["group_threshold"], cfg["ngram_range"])
    df["split"] = assign_splits(df, cfg["fractions"], seed)
    return df


def split_report(df: pd.DataFrame, cfg: dict) -> dict:
    sizes = df["group_id"].value_counts()
    per_intent = pd.crosstab(df["intent"], df["split"], normalize="index")
    return {
        "group_threshold": cfg["group_threshold"],
        "n_rows": int(len(df)),
        "n_groups": int(sizes.size),
        "largest_groups": [int(x) for x in sizes.head(5)],
        "singleton_share": round(float((sizes == 1).mean()), 4),
        "rows": {s: int((df["split"] == s).sum()) for s in SPLITS},
        "share": {s: round(float((df["split"] == s).mean()), 4) for s in SPLITS},
        "positive_rate": {s: round(float(df.loc[df["split"] == s, "label_escalate"].mean()), 4) for s in SPLITS},
        "per_intent_share_range": {s: [round(float(per_intent[s].min()), 4), round(float(per_intent[s].max()), 4)]
                                   for s in SPLITS},
    }


def main() -> int:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    cfg, seed = params["split"], params["seed"]
    label_cfg = load_label_config(cfg["labels"])
    schema = yaml.safe_load(open(cfg["schema"], encoding="utf-8"))
    check_against_schema(label_cfg, list(schema["intents"]))

    df = build_splits(pd.read_parquet(cfg["input"]), label_cfg, cfg, seed)
    report = split_report(df, cfg)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in SPLITS:
        part = df.loc[df["split"] == s, OUT_COLUMNS].sort_values("row_id").reset_index(drop=True)
        part.to_parquet(out_dir / f"{s}.parquet", index=False)

    rpath = Path(cfg["report"])
    rpath.parent.mkdir(parents=True, exist_ok=True)
    rpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"split: {report['rows']} rows, {report['n_groups']} groups, positive rate {report['positive_rate']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""B6: evidence that the split and KB are leak-free.

Runs against the real pipeline outputs in data/processed (produce them with `dvc repro build_kb`).
Each test is one claim a grader can check:
  - no group_id is shared between splits
  - no identical normalised query appears in two splits
  - no same-intent query pair with similarity >= group_threshold crosses splits
  - every KB entry comes from a train row; no R3 intent in the KB
  - labels equal the rules in configs/labels.yaml
  - positive rate and per-intent shares stay close to target in every split
  - same seed -> identical split
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.labels import add_labels, load_label_config, out_of_kb_intents
from src.split import OUT_COLUMNS, SPLITS, build_splits, normalise

PARAMS = yaml.safe_load(open("params.yaml", encoding="utf-8"))
CFG = PARAMS["split"]
PROCESSED = Path(CFG["output_dir"])
LABELS = load_label_config(CFG["labels"])

pytestmark = pytest.mark.skipif(
    not all((PROCESSED / f"{s}.parquet").exists() for s in (*SPLITS, "kb")),
    reason="run `dvc repro build_kb` first to create data/processed",
)


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {s: pd.read_parquet(PROCESSED / f"{s}.parquet") for s in SPLITS}


@pytest.fixture(scope="module")
def kb() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED / "kb.parquet")


# ------------------------------------------------------------------ contract
def test_contract_columns(splits):
    for s, df in splits.items():
        assert list(df.columns) == OUT_COLUMNS, s
        assert df["label_escalate"].isin([0, 1]).all()


def test_splits_partition_all_validated_rows(splits):
    validated = pd.read_parquet(CFG["input"])
    ids = pd.concat([df["row_id"] for df in splits.values()])
    assert ids.is_unique
    assert set(ids) == set(validated["row_id"])


# ------------------------------------------------------------------ leakage
def test_no_group_overlap_across_splits(splits):
    groups = {s: set(df["group_id"]) for s, df in splits.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                assert not groups[a] & groups[b], f"group_id shared between {a} and {b}"


def test_no_identical_normalised_query_across_splits(splits):
    norm = {s: set(df["query"].map(normalise)) for s, df in splits.items()}
    for a in SPLITS:
        for b in SPLITS:
            if a < b:
                shared = norm[a] & norm[b]
                assert not shared, f"{len(shared)} normalised queries in both {a} and {b}, e.g. {sorted(shared)[:3]}"


def test_no_near_duplicate_pair_crosses_splits(splits):
    """The grouping guarantee: within an intent, no train/val/test pair reaches the threshold."""
    allq = pd.concat([df.assign(split=s) for s, df in splits.items()], ignore_index=True)
    allq["query_norm"] = allq["query"].map(normalise)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=tuple(CFG["ngram_range"]), sublinear_tf=True)
    X = vec.fit_transform(allq["query_norm"])
    worst = 0.0
    for intent, idx in allq.groupby("intent").indices.items():
        sim = (X[idx] @ X[idx].T).toarray()
        sp = allq["split"].to_numpy()[idx]
        cross = sp[:, None] != sp[None, :]
        worst = max(worst, float(sim[cross].max()))
    assert worst < CFG["group_threshold"], f"cross-split pair with similarity {worst:.3f}"


def test_kb_rows_are_train_rows_only(splits, kb):
    assert set(kb["row_id"]) <= set(splits["train"]["row_id"])
    held_out = set(splits["val"]["row_id"]) | set(splits["test"]["row_id"])
    assert not set(kb["row_id"]) & held_out


def test_no_r3_intent_in_kb(kb):
    assert not set(kb["intent"]) & set(out_of_kb_intents(LABELS))


def test_kb_ids_unique_and_capped(kb):
    assert kb["kb_id"].is_unique
    assert kb["text"].is_unique
    assert kb.groupby("intent").size().max() <= PARAMS["build_kb"]["kb_per_intent"]


# ------------------------------------------------------------------ labels and balance
def test_labels_equal_yaml_rules(splits):
    for s, df in splits.items():
        expected = add_labels(df[["intent"]], LABELS)
        assert (df["label_rule"].to_numpy() == expected["label_rule"].to_numpy()).all(), s
        assert (df["label_escalate"].to_numpy() == expected["label_escalate"].to_numpy()).all(), s


def test_positive_rate_within_bounds(splits):
    for s, df in splits.items():
        rate = df["label_escalate"].mean()
        assert 0.22 <= rate <= 0.30, f"{s} positive rate {rate:.3f}"


def test_split_shares_close_to_target(splits):
    n = sum(len(df) for df in splits.values())
    for s, df in splits.items():
        assert abs(len(df) / n - CFG["fractions"][s]) < 0.01, s
    allq = pd.concat([df.assign(split=s) for s, df in splits.items()])
    per_intent = pd.crosstab(allq["intent"], allq["split"], normalize="index")
    for s in SPLITS:
        assert (per_intent[s] - CFG["fractions"][s]).abs().max() < 0.02, f"an intent is unbalanced in {s}"


# ------------------------------------------------------------------ determinism
def test_same_seed_gives_identical_split(splits):
    validated = pd.read_parquet(CFG["input"])
    again = build_splits(validated, LABELS, CFG, PARAMS["seed"])
    for s, df in splits.items():
        redo = again.loc[again["split"] == s, OUT_COLUMNS].sort_values("row_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(df, redo, check_dtype=False)

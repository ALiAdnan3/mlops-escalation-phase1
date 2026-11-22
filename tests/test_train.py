"""Unit tests for Member C's model code (no MLflow, no data files needed)."""
import numpy as np
import pandas as pd

from src.model import EscalationModel
from src.train import best_threshold, build_pipeline, run_name, score_split

CFG = {"min_df": 1, "max_features": 1000, "class_weight": "balanced"}


def _toy():
    pos = ["i want to talk to a human agent", "this is a complaint about my order",
           "payment failed twice", "please delete my account"] * 10
    neg = ["where is my order", "track my package", "change my shipping address",
           "how do i get an invoice"] * 10
    df = pd.DataFrame({"query": pos + neg, "label_escalate": [1] * len(pos) + [0] * len(neg)})
    df["label_rule"] = np.where(df["label_escalate"] == 1, "R1", "R0")
    return df


def test_best_threshold_separates_perfectly_separable_scores():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    thr = best_threshold(y, s)
    assert 0.3 < thr <= 0.7
    assert ((s >= thr).astype(int) == y).all()


def test_pipeline_fits_and_scores_above_chance():
    df = _toy()
    pipe = build_pipeline(CFG, (1, 2), 1.0, seed=42).fit(df["query"], df["label_escalate"])
    proba = pipe.predict_proba(df["query"])[:, 1]
    m = score_split(df, proba, best_threshold(df["label_escalate"], proba))
    assert m["f1"] > 0.9 and m["recall_R1"] > 0.9


def test_training_is_deterministic():
    df = _toy()
    a = build_pipeline(CFG, (1, 2), 1.0, 42).fit(df["query"], df["label_escalate"])
    b = build_pipeline(CFG, (1, 2), 1.0, 42).fit(df["query"], df["label_escalate"])
    assert np.allclose(a.predict_proba(df["query"]), b.predict_proba(df["query"]))


def test_escalation_model_accepts_dataframe_series_and_list():
    df = _toy()
    pipe = build_pipeline(CFG, (1, 1), 1.0, 42).fit(df["query"], df["label_escalate"])
    model = EscalationModel(pipe, threshold=0.5)
    qs = ["i want a human agent", "where is my order"]
    from_df = model.predict(pd.DataFrame({"query": qs}))
    assert list(from_df) == list(model.predict(qs)) == list(model.predict(pd.Series(qs)))
    assert list(from_df) == [1, 0]


def test_threshold_is_applied_by_the_registered_model():
    df = _toy()
    pipe = build_pipeline(CFG, (1, 1), 1.0, 42).fit(df["query"], df["label_escalate"])
    assert EscalationModel(pipe, 0.0).predict(df[["query"]]).all()      # everything escalates
    assert not EscalationModel(pipe, 1.01).predict(df[["query"]]).any()  # nothing escalates


def test_run_name_format():
    assert run_name((1, 2), 0.1) == "train-tfidf-lr-ng12-C0.1"
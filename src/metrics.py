"""Shared metric functions (Member B; C imports these so baseline and model are scored identically).

Retrieval (in-KB queries only - R3 queries have no correct answer in the KB):
  intent_hit_at_k  share of queries with a same-intent KB entry in the top k
  mrr              mean of 1 / rank of the first same-intent KB entry (full ranking)
Escalation (all queries):
  precision, recall, f1 at the chosen decision rule; pr_auc = average precision of a continuous score
MLflow metric names cannot contain '@', so intent_hit@1 is logged as intent_hit_at_1.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


def first_hit_ranks(ranked_intents: np.ndarray, true_intents: np.ndarray) -> np.ndarray:
    """ranked_intents: (n_queries, n_kb) KB intents sorted best-first. Returns 1-based rank of the
    first correct entry per query (n_kb + 1 if none, which contributes ~0 to MRR)."""
    hits = ranked_intents == np.asarray(true_intents)[:, None]
    any_hit = hits.any(axis=1)
    return np.where(any_hit, hits.argmax(axis=1) + 1, ranked_intents.shape[1] + 1)


def retrieval_metrics(ranks: np.ndarray, ks=(1, 5)) -> dict[str, float]:
    out = {f"intent_hit_at_{k}": float(np.mean(ranks <= k)) for k in ks}
    out["mrr"] = float(np.mean(1.0 / ranks))
    return out


def escalation_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f),
            "pr_auc": float(average_precision_score(y_true, y_score))}


def recall_by_rule(rules, y_pred) -> dict[str, float]:
    """Share of escalation-worthy queries caught, per rule (R1/R2/R3). Shows WHERE a model fails."""
    rules, y_pred = np.asarray(rules), np.asarray(y_pred)
    return {f"recall_{r}": float(y_pred[rules == r].mean()) for r in ("R1", "R2", "R3") if (rules == r).any()}

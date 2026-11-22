"""The registered model (Member C): a text pipeline plus the decision threshold tuned on val.

Lives in its own module (not in train.py) so pickles reference `src.model.EscalationModel`
instead of `__main__`, and `mlflow.pyfunc.load_model` works from any process that can import `src`.

Input : a pandas DataFrame with a `query` column (or a list / Series of query strings)
Output: predict() -> 0/1 (1 = escalate to a human); predict_proba() -> P(escalate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class EscalationModel:
    def __init__(self, pipeline, threshold: float = 0.5):
        self.pipeline = pipeline
        self.threshold = float(threshold)

    @staticmethod
    def _texts(X) -> pd.Series:
        if isinstance(X, pd.DataFrame):
            return X["query"].astype(str)
        return pd.Series(list(X), dtype=str)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(self._texts(X))

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
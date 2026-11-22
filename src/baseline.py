"""Stage `baseline` (Member B): run #0, the bar the primary model must beat.

Retrieval : BM25 (rank_bm25) over the KB answer texts; query and KB are normalised the same way
            the split normalises queries (lowercase, no {{placeholders}}, digits or punctuation).
Escalation: escalate if the normalised query matches `keyword_regex` (explicit human / complaint
            words) OR the retrieval confidence is below tau (weak match -> probably out of KB).
            confidence = top-1 BM25 score / number of query tokens. Raw BM25 grows with query
            length; per-token scoring separates out-of-KB queries better (val AUC 0.65 vs 0.58).
            tau is tuned on VAL only (maximise F1 over quantiles of val top-1 scores); TEST is
            scored once with that tau.
Outputs   : reports/baseline_metrics.json (DVC metrics) and an MLflow run `baseline-bm25`.

MLflow target: DagsHub when credentials are available (MLFLOW_TRACKING_USERNAME/PASSWORD, or
DAGSHUB_USERNAME + DAGSHUB_TOKEN in the environment or .env); otherwise a local sqlite:///mlflow.db
so a grader without an account can still run `dvc repro`. Team runs must go to DagsHub.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from rank_bm25 import BM25Okapi

from src.metrics import escalation_metrics, first_hit_ranks, recall_by_rule, retrieval_metrics
from src.split import normalise

SPLITS_SCORED = ("val", "test")


# ------------------------------------------------------------------ model
def tokenize(text: str) -> list[str]:
    return normalise(text).split()


class BM25Baseline:
    def __init__(self, kb: pd.DataFrame, k1: float, b: float, keyword_regex: str):
        self.kb_intents = kb["intent"].to_numpy()
        self.bm25 = BM25Okapi([tokenize(t) for t in kb["text"]], k1=k1, b=b)
        self.keywords = re.compile(keyword_regex, re.IGNORECASE)

    def score(self, queries: pd.Series) -> dict[str, np.ndarray]:
        tokens = [tokenize(q) for q in queries]
        scores = np.vstack([self.bm25.get_scores(t) for t in tokens])
        n_tokens = np.array([max(len(t), 1) for t in tokens])
        order = np.argsort(-scores, axis=1, kind="stable")          # best-first, ties by kb_id
        return {
            "ranked_intents": self.kb_intents[order],
            "top1": scores[np.arange(len(scores)), order[:, 0]],
            "confidence": scores[np.arange(len(scores)), order[:, 0]] / n_tokens,
            "keyword": np.array([bool(self.keywords.search(normalise(q))) for q in queries]),
        }

    @staticmethod
    def decide(out: dict, tau: float) -> np.ndarray:
        return (out["keyword"] | (out["confidence"] < tau)).astype(int)

    @staticmethod
    def escalation_score(out: dict) -> np.ndarray:
        """Continuous score for PR-AUC: keyword hits rank first, then weaker BM25 matches."""
        return out["keyword"] * 1e6 - out["confidence"]


def tune_tau(out: dict, y: np.ndarray, n_quantiles: int) -> float:
    candidates = np.unique(np.quantile(out["confidence"], np.linspace(0, 1, n_quantiles)))
    candidates = np.concatenate([[-np.inf], candidates, [np.inf]])
    f1s = [escalation_metrics(y, BM25Baseline.decide(out, t), BM25Baseline.escalation_score(out))["f1"]
           for t in candidates]
    return float(candidates[int(np.argmax(f1s))])     # first best: deterministic


def evaluate(df: pd.DataFrame, out: dict, tau: float) -> dict[str, float]:
    y = df["label_escalate"].to_numpy()
    pred = BM25Baseline.decide(out, tau)
    in_kb = (df["label_rule"] != "R3").to_numpy()
    ranks = first_hit_ranks(out["ranked_intents"][in_kb], df["intent"].to_numpy()[in_kb])
    return {**retrieval_metrics(ranks), **escalation_metrics(y, pred, BM25Baseline.escalation_score(out)),
            **recall_by_rule(df["label_rule"], pred)}


# ------------------------------------------------------------------ provenance + MLflow
def file_md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def data_version(paths: list[Path]) -> str:
    """Hash of the processed files' md5s (the same md5s dvc.lock records)."""
    joined = "".join(f"{p.name}:{file_md5(p)};" for p in sorted(paths))
    return hashlib.md5(joined.encode()).hexdigest()[:12]


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def load_dotenv(path: str = ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines); never overrides variables already set."""
    p = Path(path)
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() and v.strip():
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def resolve_tracking_uri(dagshub_uri: str) -> str:
    load_dotenv()
    if os.environ.get("MLFLOW_TRACKING_URI"):
        return os.environ["MLFLOW_TRACKING_URI"]
    user = os.environ.get("MLFLOW_TRACKING_USERNAME") or os.environ.get("DAGSHUB_USERNAME")
    token = os.environ.get("MLFLOW_TRACKING_PASSWORD") or os.environ.get("DAGSHUB_TOKEN")
    if user and token:
        os.environ["MLFLOW_TRACKING_USERNAME"], os.environ["MLFLOW_TRACKING_PASSWORD"] = user, token
        return dagshub_uri
    print("baseline: no DagsHub credentials found -> logging to local sqlite:///mlflow.db "
          "(fine for graders; team runs must go to DagsHub)", file=sys.stderr)
    return "sqlite:///mlflow.db"


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)[:500]
    return out


def log_to_mlflow(params: dict, cfg: dict, metrics: dict, tau: float, dv: str, metrics_path: Path) -> str:
    import mlflow
    uri = resolve_tracking_uri(cfg["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow_experiment"])
    with mlflow.start_run(run_name=cfg["run_name"]) as run:
        mlflow.log_params(flatten(params))
        mlflow.log_params({"tau": tau, "data_version": dv})
        mlflow.set_tags({"git_sha": git_sha(), "data_version": dv, "stage": "baseline",
                         "model": "bm25+threshold+keywords", "owner": "member-B"})
        for split in SPLITS_SCORED:
            mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics[split].items()})
        mlflow.log_artifact(str(metrics_path))
        print(f"baseline: MLflow run {run.info.run_id} -> {uri}")
    return uri


# ------------------------------------------------------------------ main
def main() -> int:
    # mlflow prints an emoji when a run ends; on Windows a piped stdout is cp1252 and crashes on it
    sys.stdout.reconfigure(encoding="utf-8")
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    cfg = params["baseline"]
    processed = Path(cfg["splits_dir"])
    kb = pd.read_parquet(cfg["kb"])
    data = {s: pd.read_parquet(processed / f"{s}.parquet") for s in SPLITS_SCORED}

    model = BM25Baseline(kb, cfg["bm25_k1"], cfg["bm25_b"], cfg["keyword_regex"])
    outs = {s: model.score(df["query"]) for s, df in data.items()}
    tau = tune_tau(outs["val"], data["val"]["label_escalate"].to_numpy(), cfg["tau_quantiles"])
    metrics = {s: evaluate(data[s], outs[s], tau) for s in SPLITS_SCORED}

    dv = data_version([processed / f"{s}.parquet" for s in ("train", "val", "test", "kb")])
    report = {"run_name": cfg["run_name"], "tau": tau, "data_version": dv,
              **{s: {k: round(v, 6) for k, v in m.items()} for s, m in metrics.items()}}
    mpath = Path(cfg["metrics"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))

    log_to_mlflow(params, cfg, metrics, tau, dv, mpath)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Stage `train` (Member C): train, evaluate and register the escalation classifier.

Model     : TF-IDF (word n-grams of the normalised query) + logistic regression. Text only, so it
            needs no new packages. Retrieval features / embeddings are a later upgrade.
Sweep     : ngram_range x C from params.yaml `train.sweep`; every combination is one MLflow run.
Selection : champion = best val F1. The decision threshold is tuned on VAL only (same rule as the
            baseline's tau). TEST is scored once, for the champion only.
Scoring   : src/metrics.py, the same code that scored `baseline-bm25`. Metrics are logged as
            val_<metric> / test_<metric> so the runs table lines up with the baseline run.
Registry  : champion is registered as `escalation-model`, stage Staging + alias `champion`.
Outputs   : reports/registered_metrics.json (DVC metrics) and MLflow runs on DagsHub when
            DAGSHUB_USERNAME + DAGSHUB_TOKEN are set (environment or .env), else local sqlite.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline

from src.metrics import escalation_metrics, recall_by_rule
from src.model import EscalationModel
from src.split import normalise


# ------------------------------------------------------------------ model + scoring
def build_pipeline(cfg: dict, ngram_range, C: float, seed: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=normalise, ngram_range=tuple(ngram_range),
                                  min_df=cfg["min_df"], max_features=cfg["max_features"],
                                  sublinear_tf=True)),
        ("clf", LogisticRegression(C=C, class_weight=cfg["class_weight"], max_iter=1000,
                                   random_state=seed)),
    ])


def best_threshold(y_true, y_score) -> float:
    """Threshold that maximises F1 on the given (val) data; first best, so deterministic."""
    p, r, t = precision_recall_curve(y_true, y_score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[int(np.argmax(f1))])


def score_split(df: pd.DataFrame, proba: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (proba >= threshold).astype(int)
    return {**escalation_metrics(df["label_escalate"].to_numpy(), pred, proba),
            **recall_by_rule(df["label_rule"], pred)}


def run_name(ngram_range, C) -> str:
    return f"train-tfidf-lr-ng{ngram_range[0]}{ngram_range[1]}-C{C:g}"


# ------------------------------------------------------------------ MLflow
def log_champion_model(mlflow, model: EscalationModel, sample: pd.DataFrame, cfg: dict,
                       dv: str, sha: str) -> str:
    """Log + register the champion inside the active run; returns the model version string."""
    from mlflow.models import infer_signature
    from mlflow.tracking import MlflowClient

    name = cfg["registered_model"]
    example = sample[["query"]].head(5)
    signature = infer_signature(example, model.predict(example))
    info = mlflow.sklearn.log_model(model, artifact_path="model", signature=signature,
                                    input_example=example, registered_model_name=name)
    client = MlflowClient()
    version = getattr(info, "registered_model_version", None)
    if version is None:   # older/other servers: look the new version up by its source run
        run_id = mlflow.active_run().info.run_id
        version = max(client.search_model_versions(f"run_id='{run_id}'"), key=lambda v: int(v.version)).version
    version = str(version)
    for k, v in {"data_version": dv, "git_sha": sha, "threshold": f"{model.threshold:.6f}"}.items():
        client.set_model_version_tag(name, version, k, v)
    # The brief wants a visible stage; the alias is a bonus and not every server supports it.
    client.transition_model_version_stage(name, version, stage="Staging")
    try:
        client.set_registered_model_alias(name, "champion", version)
    except Exception as exc:  # noqa: BLE001
        print(f"train: alias 'champion' not set ({exc}); stage Staging is set", file=sys.stderr)
    return version


# ------------------------------------------------------------------ main
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")   # mlflow prints an emoji; Windows pipes are cp1252
    import mlflow
    from src.baseline import data_version, flatten, git_sha, resolve_tracking_uri

    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    seed, cfg = params["seed"], params["train"]
    proc = Path(cfg["splits_dir"])
    train, val, test = (pd.read_parquet(proc / f"{s}.parquet") for s in ("train", "val", "test"))
    y_train, y_val = train["label_escalate"].to_numpy(), val["label_escalate"].to_numpy()

    dv = data_version([proc / f"{s}.parquet" for s in ("train", "val", "test", "kb")])
    sha = git_sha()
    uri = resolve_tracking_uri(cfg["tracking_uri"])
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(cfg["mlflow_experiment"])
    print(f"train: logging to {uri}")

    best = None   # (val_f1, run_id, pipeline, threshold, val_metrics, run_name)
    for ngram_range in cfg["sweep"]["ngram_range"]:
        for C in cfg["sweep"]["C"]:
            pipe = build_pipeline(cfg, ngram_range, C, seed).fit(train["query"], y_train)
            thr = best_threshold(y_val, pipe.predict_proba(val["query"])[:, 1])
            vm = score_split(val, pipe.predict_proba(val["query"])[:, 1], thr)
            run_params = {"model": "tfidf+logreg", "ngram_range": str(tuple(ngram_range)), "C": C,
                          "min_df": cfg["min_df"], "max_features": cfg["max_features"],
                          "class_weight": cfg["class_weight"], "threshold": round(thr, 6),
                          "seed": seed, "data_version": dv}
            with mlflow.start_run(run_name=run_name(ngram_range, C)) as run:
                mlflow.log_params(run_params)
                mlflow.set_tags({"git_sha": sha, "data_version": dv, "stage": "train",
                                 "owner": "member-C"})
                mlflow.log_metrics({f"val_{k}": v for k, v in vm.items()})
            print(f"train: {run_name(ngram_range, C)}  val_f1={vm['f1']:.4f}  thr={thr:.4f}")
            if best is None or vm["f1"] > best[0]:
                best = (vm["f1"], run.info.run_id, pipe, thr, vm, run_name(ngram_range, C))

    # ---- champion: score TEST once, register
    _, run_id, pipe, thr, vm, champion_name = best
    tm = score_split(test, pipe.predict_proba(test["query"])[:, 1], thr)
    model = EscalationModel(pipe, thr)
    report = {"run_name": champion_name,
              "registered_model": cfg["registered_model"], "data_version": dv,
              "threshold": round(thr, 6),
              "val": {k: round(v, 6) for k, v in vm.items()},
              "test": {k: round(v, 6) for k, v in tm.items()}}
    mpath = Path(cfg["metrics"])
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics({f"test_{k}": v for k, v in tm.items()})
        mlflow.set_tag("champion", "true")
        mlflow.log_artifact(str(mpath))
        version = log_champion_model(mlflow, model, val, cfg, dv, sha)
    print(f"train: registered {cfg['registered_model']} v{version} (Staging) from run {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
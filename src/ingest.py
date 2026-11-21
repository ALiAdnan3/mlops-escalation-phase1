"""Stage `ingest`: download Bitext from Hugging Face at a pinned revision -> data/raw/bitext.parquet.

The raw file is never edited afterwards; every downstream change happens in later stages.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

RAW_COLUMNS = ["instruction", "response", "category", "intent", "flags"]


def load_params(path: str = "params.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_params()["ingest"]
    revision = cfg["hf_revision"]
    if revision in ("main", "master", "", None):
        print("ingest: hf_revision must be a pinned commit SHA, not a branch name", file=sys.stderr)
        return 1

    csv_path = hf_hub_download(
        repo_id=cfg["hf_dataset"],
        filename=cfg["hf_file"],
        revision=revision,
        repo_type="dataset",
    )
    # keep_default_na=False: the literal strings are data, empty cells stay "" (validate catches them)
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        print(f"ingest: upstream file is missing columns {missing}", file=sys.stderr)
        return 1
    df = df[RAW_COLUMNS]

    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"ingest: wrote {len(df)} rows -> {out} (revision {revision[:10]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

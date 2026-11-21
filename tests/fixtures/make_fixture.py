"""Regenerate tests/fixtures/good_sample.csv: 2 real rows per intent from data/raw (seed from params.yaml).

Run from the repo root:  python -m tests.fixtures.make_fixture
"""
from pathlib import Path

import pandas as pd
import yaml

params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
raw = pd.read_parquet(params["ingest"]["output"])
sample = (
    raw.drop_duplicates(["instruction", "intent"])
    .groupby("intent", group_keys=False)
    .apply(lambda g: g.sample(2, random_state=params["seed"]))
    .sort_values(["intent", "instruction"])
    .reset_index(drop=True)
)
out = Path(__file__).with_name("good_sample.csv")
sample.to_csv(out, index=False)
print(f"wrote {len(sample)} rows -> {out}")

"""Stage `build_kb` (Member B): the knowledge base the bot retrieves answers from.

Reads  data/processed/train.parquet (which rows are allowed) and data/interim/validated.parquet
       (the response text for those rows)
Writes data/processed/kb.parquet with columns kb_id, intent, text, row_id

Leakage rules enforced here (and re-checked in tests/test_split.py):
- only TRAIN rows contribute KB entries; val/test answers never enter the KB;
- R3 intents are dropped, so those queries have no correct answer in the KB (the bot should escalate);
- duplicate response texts are removed, then each intent is capped at kb_per_intent entries
  (a seeded sample, so the KB is identical on every run).
`row_id` is kept as provenance so tests can prove every KB entry came from a train row.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import yaml

from src.labels import load_label_config, out_of_kb_intents

KB_COLUMNS = ["kb_id", "intent", "text", "row_id"]


def build_kb(train: pd.DataFrame, validated: pd.DataFrame, excluded_intents: list[str],
             kb_per_intent: int, seed: int) -> pd.DataFrame:
    rows = validated.loc[validated["row_id"].isin(train["row_id"]), ["row_id", "intent", "response"]]
    rows = rows[~rows["intent"].isin(excluded_intents)]
    rows = rows.sort_values("row_id").drop_duplicates("response", keep="first")
    parts = []
    for intent, grp in rows.groupby("intent", sort=True):
        n = min(kb_per_intent, len(grp))
        parts.append(grp.sample(n=n, random_state=seed).sort_values("row_id"))
    kb = pd.concat(parts, ignore_index=True).rename(columns={"response": "text"})
    kb.insert(0, "kb_id", pd.RangeIndex(len(kb), dtype="int64"))
    return kb[KB_COLUMNS]


def main() -> int:
    params = yaml.safe_load(open("params.yaml", encoding="utf-8"))
    cfg, seed = params["build_kb"], params["seed"]
    excluded = out_of_kb_intents(load_label_config(cfg["labels"]))

    kb = build_kb(pd.read_parquet(cfg["train"]), pd.read_parquet(cfg["validated"]),
                  excluded, cfg["kb_per_intent"], seed)
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    kb.to_parquet(out, index=False)
    print(f"build_kb: {len(kb)} entries over {kb['intent'].nunique()} intents -> {out} (excluded {excluded})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Escalation labels (Member B): map a Bitext intent to `label_escalate` (0/1) and `label_rule` (R0-R3).

The rules live in configs/labels.yaml so they are data, not code: split.py and the tests read the
same file, which is what makes "labels equal the YAML rule" testable.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

ESCALATE_RULES = ("R1", "R2", "R3")
OUT_OF_KB_RULE = "R3"


def load_label_config(path: str | Path = "configs/labels.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def intent_to_rule(cfg: dict) -> dict[str, str]:
    """Invert the YAML: {intent: rule}. Fails loudly if an intent is listed under two rules."""
    mapping: dict[str, str] = {}
    for rule, intents in cfg["rules"].items():
        for intent in intents:
            if intent in mapping:
                raise ValueError(f"intent {intent!r} listed under both {mapping[intent]} and {rule}")
            mapping[intent] = rule
    return mapping


def check_against_schema(cfg: dict, known_intents: list[str]) -> None:
    """Every intent named in labels.yaml must exist in A's frozen schema (catches typos)."""
    unknown = sorted(set(intent_to_rule(cfg)) - set(known_intents))
    if unknown:
        raise ValueError(f"labels.yaml names intents not in the schema: {unknown}")


def out_of_kb_intents(cfg: dict) -> list[str]:
    return list(cfg["rules"][OUT_OF_KB_RULE])


def add_labels(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Return a copy of df with label_rule and label_escalate columns."""
    mapping = intent_to_rule(cfg)
    out = df.copy()
    out["label_rule"] = out["intent"].map(mapping).fillna(cfg["default_rule"])
    out["label_escalate"] = out["label_rule"].isin(ESCALATE_RULES).astype("int8")
    return out

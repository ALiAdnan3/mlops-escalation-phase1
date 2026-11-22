"""B1: the label function covers all 27 intents and matches the frozen rules."""
import pandas as pd
import pytest
import yaml

from src.labels import add_labels, check_against_schema, intent_to_rule, load_label_config

CFG = load_label_config()
SCHEMA_INTENTS = list(yaml.safe_load(open("configs/schema.yaml", encoding="utf-8"))["intents"])


def test_all_listed_intents_exist_in_schema():
    check_against_schema(CFG, SCHEMA_INTENTS)


def test_covers_all_27_intents_with_7_positives():
    df = add_labels(pd.DataFrame({"intent": SCHEMA_INTENTS}), CFG)
    assert len(df) == 27
    assert df["label_rule"].notna().all()
    assert df["label_escalate"].sum() == 7
    assert set(df["label_rule"]) == {"R0", "R1", "R2", "R3"}


@pytest.mark.parametrize("intent,rule,esc", [
    ("contact_human_agent", "R1", 1), ("complaint", "R1", 1),
    ("payment_issue", "R2", 1), ("delete_account", "R2", 1),
    ("switch_account", "R3", 1), ("newsletter_subscription", "R3", 1), ("review", "R3", 1),
    ("track_order", "R0", 0), ("get_refund", "R0", 0),
])
def test_specific_intents(intent, rule, esc):
    row = add_labels(pd.DataFrame({"intent": [intent]}), CFG).iloc[0]
    assert (row["label_rule"], row["label_escalate"]) == (rule, esc)


def test_duplicate_intent_across_rules_is_rejected():
    bad = {"rules": {"R1": ["complaint"], "R2": ["complaint"]}, "default_rule": "R0"}
    with pytest.raises(ValueError, match="both"):
        intent_to_rule(bad)


def test_typo_in_labels_is_caught():
    bad = {"rules": {"R1": ["contact_human_agnet"]}, "default_rule": "R0"}
    with pytest.raises(ValueError, match="not in the schema"):
        check_against_schema(bad, SCHEMA_INTENTS)

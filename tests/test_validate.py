"""Validation gate tests: the good fixture passes; each deliberately corrupted copy fails with a clear message."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.validate import add_row_id, validate_df

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "good_sample.csv"


@pytest.fixture(scope="module")
def schema_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "schema.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def vcfg() -> dict:
    cfg = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))["validate"]
    # The fixture is tiny (2 rows/intent), so relax only the size thresholds; every other rule is real.
    return {**cfg, "min_rows_per_intent": 2, "min_total_rows": 54}


@pytest.fixture()
def good() -> pd.DataFrame:
    return pd.read_csv(FIXTURE, dtype=str, keep_default_na=False)


def failed_checks(report: dict) -> set[str]:
    return {e["check"] for e in report["errors"]}


def assert_fails(report: dict, check: str, text: str) -> None:
    assert report["passed"] is False, "validation should have failed"
    assert check in failed_checks(report), f"expected {check!r}, got {failed_checks(report)}"
    msgs = " | ".join(e["message"] for e in report["errors"] if e["check"] == check)
    assert text in msgs, f"message {msgs!r} should mention {text!r}"


# ------------------------------------------------------------------ good path
def test_good_fixture_passes(good, schema_cfg, vcfg):
    report = validate_df(good, schema_cfg, vcfg)
    assert report["passed"], report["errors"]
    assert report["stats"]["n_intents"] == 27
    assert report["stats"]["n_categories"] == 11


def test_row_id_is_stable_integer(good):
    a, b = add_row_id(good), add_row_id(good)
    assert a["row_id"].dtype == "int64"
    assert a["row_id"].is_unique and a["row_id"].tolist() == list(range(len(good)))
    pd.testing.assert_frame_equal(a, b)


# ------------------------------------------------------------------ corrupted copies
def test_missing_column_fails(good, schema_cfg, vcfg):
    report = validate_df(good.drop(columns=["response"]), schema_cfg, vcfg)
    assert_fails(report, "required_columns", "response")


def test_null_intent_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    bad.loc[0, "intent"] = None
    assert_fails(validate_df(bad, schema_cfg, vcfg), "not_nullable", "intent")


def test_unknown_intent_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    bad.loc[0, "intent"] = "teleport_order"
    report = validate_df(bad, schema_cfg, vcfg)
    assert_fails(report, "known_values", "intent must be one of the 27 known values")


def test_unknown_category_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    bad.loc[0, "category"] = "MARKETING"
    assert_fails(validate_df(bad, schema_cfg, vcfg), "known_values", "category must be one of the 11 known values")


@pytest.mark.parametrize("value", ["", "    "])
def test_empty_text_fails(good, schema_cfg, vcfg, value):
    bad = good.copy()
    bad.loc[0, "instruction"] = value
    assert_fails(validate_df(bad, schema_cfg, vcfg), "not_blank", "instruction must not be empty/whitespace")


def test_too_long_response_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    bad.loc[0, "response"] = "x" * (vcfg["max_response_len"] + 1)
    assert_fails(validate_df(bad, schema_cfg, vcfg), "length_bounds", f"response length must be in [3, {vcfg['max_response_len']}] chars")


def test_wrong_dtype_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    bad["intent"] = np.arange(len(bad), dtype="int64")
    assert_fails(validate_df(bad, schema_cfg, vcfg), "dtype('object')", "intent")


def test_intent_category_mismatch_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    i = bad.index[bad["intent"] == "track_order"][0]
    bad.loc[i, "category"] = "REFUND"  # valid category, wrong for this intent
    assert_fails(validate_df(bad, schema_cfg, vcfg), "intent_category_consistent", "intent does not belong to its category")


def test_duplicate_rate_fails(good, schema_cfg, vcfg):
    bad = pd.concat([good, good.iloc[:20]], ignore_index=True)  # 20/74 = 27% dupes
    assert_fails(validate_df(bad, schema_cfg, vcfg), "duplicate_rate", f"duplicate rate exceeds {vcfg['max_duplicate_rate']}")


def test_missing_intent_fails(good, schema_cfg, vcfg):
    bad = good[good["intent"] != "complaint"]
    report = validate_df(bad, schema_cfg, {**vcfg, "min_total_rows": 1})
    assert_fails(report, "min_rows_per_intent", "fewer than 2 rows (or is missing)")


def test_bad_flags_fails(good, schema_cfg, vcfg):
    bad = good.copy()
    bad.loc[0, "flags"] = "b-l"
    assert_fails(validate_df(bad, schema_cfg, vcfg), "flag_letters", "flags must be upper-case letter codes")


# ------------------------------------------------------------------ the stage itself halts
def test_stage_exits_nonzero_on_bad_file(tmp_path, good):
    """End-to-end: the CLI stage must exit non-zero, write a report, and not write an output file."""
    bad = good.copy()
    bad.loc[0, "intent"] = "teleport_order"
    bad.to_parquet(tmp_path / "raw.parquet", index=False)

    params = yaml.safe_load((ROOT / "params.yaml").read_text(encoding="utf-8"))
    params["validate"].update(
        input=str(tmp_path / "raw.parquet"), output=str(tmp_path / "validated.parquet"),
        report=str(tmp_path / "validation.json"), schema=str(ROOT / "configs" / "schema.yaml"),
        min_rows_per_intent=2, min_total_rows=54,
    )
    (tmp_path / "params.yaml").write_text(yaml.safe_dump(params), encoding="utf-8")

    proc = subprocess.run([sys.executable, str(ROOT / "src" / "validate.py")], cwd=tmp_path,
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "FAILED" in proc.stderr and "intent" in proc.stderr
    assert (tmp_path / "validation.json").exists()
    assert not (tmp_path / "validated.parquet").exists()

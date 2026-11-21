"""Stage `validate`: schema + quality gate between ingest and split.

Reads  data/raw/bitext.parquet
Writes data/interim/validated.parquet (raw columns + stable integer `row_id`) and reports/validation.json.
Exits non-zero on any failure so `dvc repro` stops before training ever sees bad data.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pandera as pa
import yaml

RAW_COLUMNS = ["instruction", "response", "category", "intent", "flags"]
TEXT_COLUMNS = ["instruction", "response"]


def load_yaml(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- checks
def _is_str(s: pd.Series) -> pd.Series:
    return s.map(lambda v: isinstance(v, str))


def _not_blank(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.len() > 0


def _len_between(lo: int, hi: int):
    def check(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.len().between(lo, hi)
    return check


def build_schema(schema_cfg: dict, vcfg: dict) -> pa.DataFrameSchema:
    intents: dict[str, str] = schema_cfg["intents"]
    categories: list[str] = schema_cfg["categories"]
    lo = vcfg["min_text_len"]
    max_len = {"instruction": vcfg["max_instruction_len"], "response": vcfg["max_response_len"]}

    def text_col(name: str) -> pa.Column:
        return pa.Column(
            object, nullable=False,
            checks=[
                pa.Check(_is_str, name="is_string", error=f"{name} must contain strings"),
                pa.Check(_not_blank, name="not_blank", error=f"{name} must not be empty/whitespace"),
                pa.Check(_len_between(lo, max_len[name]), name="length_bounds",
                         error=f"{name} length must be in [{lo}, {max_len[name]}] chars"),
            ],
        )

    def label_col(name: str, allowed: list[str]) -> pa.Column:
        return pa.Column(
            object, nullable=False,
            checks=[
                pa.Check(_is_str, name="is_string", error=f"{name} must contain strings"),
                pa.Check(lambda s: s.isin(allowed), name="known_values",
                         error=f"{name} must be one of the {len(allowed)} known values"),
            ],
        )

    # ---- dataset-level checks (return bool or a row-wise bool Series) ----
    def intent_matches_category(df: pd.DataFrame) -> pd.Series:
        return df["intent"].map(intents).eq(df["category"]) | ~df["intent"].isin(intents)

    def duplicate_rate_ok(df: pd.DataFrame) -> bool:
        return df.duplicated(["instruction", "intent"]).mean() <= vcfg["max_duplicate_rate"]

    def min_rows_per_intent_ok(df: pd.DataFrame) -> bool:
        counts = df["intent"].value_counts()
        return all(counts.get(i, 0) >= vcfg["min_rows_per_intent"] for i in intents)

    def min_total_rows_ok(df: pd.DataFrame) -> bool:
        return len(df) >= vcfg["min_total_rows"]

    return pa.DataFrameSchema(
        columns={
            "instruction": text_col("instruction"),
            "response": text_col("response"),
            "category": label_col("category", categories),
            "intent": label_col("intent", list(intents)),
            "flags": pa.Column(object, nullable=False, checks=[
                pa.Check(_is_str, name="is_string", error="flags must contain strings"),
                pa.Check(lambda s: s.astype(str).str.fullmatch(r"[A-Z]+"), name="flag_letters",
                         error="flags must be upper-case letter codes"),
            ]),
        },
        checks=[
            pa.Check(intent_matches_category, name="intent_category_consistent",
                     error="intent does not belong to its category"),
            pa.Check(duplicate_rate_ok, name="duplicate_rate",
                     error=f"(instruction, intent) duplicate rate exceeds {vcfg['max_duplicate_rate']}"),
            pa.Check(min_rows_per_intent_ok, name="min_rows_per_intent",
                     error=f"an intent has fewer than {vcfg['min_rows_per_intent']} rows (or is missing)"),
            pa.Check(min_total_rows_ok, name="min_total_rows",
                     error=f"fewer than {vcfg['min_total_rows']} rows"),
        ],
        strict=False,  # extra columns are tolerated but dropped on output
    )


# --------------------------------------------------------------------------- core
def dataset_stats(df: pd.DataFrame) -> dict:
    stats: dict = {"n_rows": int(len(df)), "columns": list(map(str, df.columns))}
    if "intent" in df:
        stats["n_intents"] = int(df["intent"].nunique())
        stats["rows_per_intent"] = {str(k): int(v) for k, v in df["intent"].value_counts().sort_index().items()}
    if "category" in df:
        stats["n_categories"] = int(df["category"].nunique())
    if {"instruction", "intent"} <= set(df.columns):
        stats["duplicate_rate"] = round(float(df.duplicated(["instruction", "intent"]).mean()), 4)
    for c in TEXT_COLUMNS:
        if c in df and len(df):
            lens = df[c].astype(str).str.len()
            stats[f"{c}_len"] = {"min": int(lens.min()), "max": int(lens.max()), "mean": round(float(lens.mean()), 1)}
    return stats


def validate_df(df: pd.DataFrame, schema_cfg: dict, vcfg: dict) -> dict:
    """Validate `df`. Returns a report dict with `passed` and a list of human-readable `errors`."""
    report: dict = {"passed": True, "errors": [], "stats": dataset_stats(df)}

    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:  # Pandera would also flag this, but say it plainly and stop early
        report["passed"] = False
        report["errors"].append({"check": "required_columns", "column": None,
                                 "n_failures": len(missing),
                                 "message": f"missing required column(s): {missing}", "examples": missing})
        return report

    schema = build_schema(schema_cfg, vcfg)
    # Pandera reports a check by its `error` text; map that back to our short, stable check name.
    all_checks = list(schema.checks) + [c for col in schema.columns.values() for c in col.checks]
    name_of = {c.error: c.name for c in all_checks if c.error}

    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        report["passed"] = False
        fc = exc.failure_cases
        for (col, check), grp in fc.groupby(["column", "check"], dropna=False, sort=False):
            col = None if pd.isna(col) else str(col)
            check = str(check)
            report["errors"].append({
                "check": name_of.get(check, check),   # built-ins: not_nullable, dtype('object'), ...
                "column": col,
                "n_failures": int(len(grp)),
                "message": f"[{col or 'dataframe'}] {check}",
                "examples": [str(v)[:120] for v in grp["failure_case"].head(5)],
            })
    return report


def add_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """row_id = position in the immutable raw file (pinned revision), so it is stable across runs."""
    out = df[RAW_COLUMNS].reset_index(drop=True).copy()
    out.insert(0, "row_id", pd.RangeIndex(len(out), dtype="int64"))
    return out


def main() -> int:
    params = load_yaml("params.yaml")
    vcfg = params["validate"]
    schema_cfg = load_yaml(vcfg["schema"])

    df = pd.read_parquet(vcfg["input"])
    report = validate_df(df, schema_cfg, vcfg)

    report_path = Path(vcfg["report"])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n": same bytes on every OS, so the hash in dvc.lock matches a fresh git checkout
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")

    out = Path(vcfg["output"])
    if not report["passed"]:
        out.unlink(missing_ok=True)  # never leave a stale "validated" file behind
        print(f"validate: FAILED with {len(report['errors'])} error(s), see {report_path}", file=sys.stderr)
        for e in report["errors"]:
            print(f"  - {e['message']}: {e['n_failures']} failure(s), e.g. {e['examples'][:3]}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    add_row_id(df).to_parquet(out, index=False)
    print(f"validate: PASSED, {len(df)} rows -> {out}; report -> {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

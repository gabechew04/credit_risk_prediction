"""Write a cleaned copy of credit_risk/datasets/loan.csv.

Cleaning steps
--------------
1. Drop identifier columns (loan_id, customer_id), redundant columns
   (funded_amount, description, issue_date) and uninformative columns
   (notes, pymnt_plan) - see DROP_COLUMNS for why each goes.
2. Normalise `type` to lowercase with whitespace collapsed to underscores, then
   merge the split levels: Individual/INDIVIDUAL -> individual, and
   Joint App/JOINT -> joint.

No other column is modified.

A second output, loan_status_recoded.csv, is derived from the cleaned dataset by
collapsing loan_status to a resolved binary outcome: `Current` and `In Grace
Period` rows (no resolved outcome yet) are dropped, `Fully Paid` is kept as-is,
and everything else (Charged Off, Late 16-30/31-120 days, Default) becomes
`Missed Payment`. Bar and pie charts of the recoded column are saved to
documentation/figures/.

Requires the `data_science` conda environment (pandas 3.0.3, numpy 2.5.1):

    conda activate data_science
    python credit_risk/clean_loan_dataset.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/clean_loan_dataset.py

To point at a different source file or a row-capped sample, edit
SOURCE_PATH / OUTPUT_PATH / SAMPLE below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from univariate_analysis import FIG_DIR, bar_chart, pie_chart

DATA_DIR = Path(__file__).resolve().parent / "datasets"
SOURCE_PATH = DATA_DIR / "loan.csv"
OUTPUT_PATH = DATA_DIR / "loan_cleaned.csv"
RECODED_OUTPUT_PATH = DATA_DIR / "loan_status_recoded.csv"
SAMPLE = None  # int to read only the first N rows, None for the full file

# Column -> reason it is dropped.
DROP_COLUMNS = {
    "loan_id": "identifier (row index 1..N, no predictive content)",
#    "customer_id": "identifier (hashed bytes, unique per row)",
    "funded_amount": "duplicate of loan_amount (identical on every row)",
    "description": "duplicate of purpose (free-text restatement, 5,631 dirty levels)",
    "issue_date": "duplicate of issue_d (same date, verbose format)",
    "notes": "constant ('desc' on every row, zero information)",
    "pymnt_plan": "near-constant (99.99% False, only 40 True rows)",
}

# Columns normalised to lowercase_with_underscores.
NORMALISE_COLUMNS = ["type"]

# Applied after casing normalisation: `Joint App` and `JOINT` both mean a joint
# application, so they fold into one level.
TYPE_ALIASES = {"joint_app": "joint"}

# loan_status values with no resolved outcome yet - dropped from the recoded set.
DROP_LOAN_STATUS = {"Current", "In Grace Period"}

# Remaining loan_status values folded into a binary outcome; `Fully Paid` is
# left as-is (not listed here) and everything else becomes `Missed Payment`.
LOAN_STATUS_ALIASES = {
    "Charged Off": "Missed Payment",
    "Late (31-120 days)": "Missed Payment",
    "Late (16-30 days)": "Missed Payment",
    "Default": "Missed Payment",
}


def normalise_text(s: pd.Series) -> pd.Series:
    """Lowercase, trim, and replace runs of whitespace with single underscores."""
    return (
        s.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # 1. Drop identifier and redundant columns.
    present = [c for c in DROP_COLUMNS if c in out.columns]
    out = out.drop(columns=present)

    # 2. Normalise casing/whitespace, then fold the joint-application aliases.
    for col in NORMALISE_COLUMNS:
        if col in out.columns:
            out[col] = normalise_text(out[col])
    if "type" in out.columns:
        out["type"] = out["type"].replace(TYPE_ALIASES)

    return out


def recode_loan_status(df: pd.DataFrame) -> pd.DataFrame:
    """Drop in-progress loan statuses and collapse the rest to a binary outcome.

    `Current` and `In Grace Period` loans have no resolved outcome yet, so those
    rows are dropped rather than labelled. Everything left that isn't
    `Fully Paid` (Charged Off, Late 16-30/31-120 days, Default) becomes
    `Missed Payment`.
    """
    out = df[~df["loan_status"].isin(DROP_LOAN_STATUS)].copy()
    out["loan_status"] = out["loan_status"].replace(LOAN_STATUS_ALIASES)
    return out


def main() -> None:
    print(f"Reading {SOURCE_PATH} ...")
    df = pd.read_csv(SOURCE_PATH, nrows=SAMPLE, low_memory=False)
    print(f"  source: {df.shape[0]:,} rows x {df.shape[1]} columns")

    before_type = df["type"].value_counts() if "type" in df.columns else None

    cleaned = clean(df)

    print("\nDropped columns:")
    for col, reason in DROP_COLUMNS.items():
        mark = "-" if col in df.columns else "  (absent)"
        print(f"  {mark} {col:<16} {reason}")

    if before_type is not None:
        print("\n`type` before:")
        print(before_type.to_string())
        after_type = cleaned["type"].value_counts()
        print("\n`type` after:")
        print(pd.DataFrame(
            {"count": after_type, "pct": (100 * after_type / len(cleaned)).round(2)}
        ).to_string())

    print(f"\n  cleaned: {cleaned.shape[0]:,} rows x {cleaned.shape[1]} columns")
    print(f"  columns: {list(cleaned.columns)}")

    # This script only drops columns and rewrites `type` in place.
    assert len(cleaned) == len(df), "row count changed during cleaning"
    untouched = [c for c in cleaned.columns if c not in NORMALISE_COLUMNS]
    for col in untouched:
        assert cleaned[col].equals(df[col]), f"{col} was modified unexpectedly"

    cleaned.to_csv(OUTPUT_PATH, index=False)
    size_mb = OUTPUT_PATH.stat().st_size / 1024**2
    print(f"\nWrote {OUTPUT_PATH}  ({size_mb:.1f} MB)")

    # --- loan_status recode: binary Fully Paid vs Missed Payment outcome ---
    # print("\n`loan_status` before recode:")
    # print(cleaned["loan_status"].value_counts().to_string())

    # #recoded = recode_loan_status(cleaned)

    # print("\n`loan_status` after recode (Current/In Grace Period dropped):")
    # print(recoded["loan_status"].value_counts().to_string())
    # print(f"  rows: {len(cleaned):,} -> {len(recoded):,} ({len(cleaned) - len(recoded):,} dropped)")

    # recoded.to_csv(RECODED_OUTPUT_PATH, index=False)
    # size_mb = RECODED_OUTPUT_PATH.stat().st_size / 1024**2
    # print(f"\nWrote {RECODED_OUTPUT_PATH}  ({size_mb:.1f} MB)")



if __name__ == "__main__":
    main()

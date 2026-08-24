"""Column-level analysis of credit_risk/datasets/loan.csv.

Produces, for every column:
  1. Variable types   - pandas dtype, an inferred semantic type, and cardinality.
  2. Missing values   - count and percentage of nulls (blank/whitespace strings
                        and common null placeholders are treated as missing).
  3. Summary          - describe() for numeric/datetime columns, value counts and
                        top categories for object/categorical columns.

Requires the `data_science` conda environment (pandas 3.0.3, numpy 2.5.1):

    conda activate data_science
    python credit_risk/loan_column_analysis.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/loan_column_analysis.py

To analyse a different file or a row-capped sample, edit DATA_PATH / SAMPLE below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "datasets" / "master.csv"
SAMPLE = None  # int to read only the first N rows, None for the full file

# # Strings that mean "no value" but are not read as NaN by default.
# NULL_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?", "unknown", "missing"}

# A column with few distinct values relative to its length is treated as
# categorical even when stored as text or as an integer-valued float.
CATEGORICAL_MAX_UNIQUE = 50
CATEGORICAL_MAX_RATIO = 0.05

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def load(path: Path, sample: int | None) -> pd.DataFrame:
    """Read the CSV, letting pandas infer dtypes, with an optional row cap."""
    df = pd.read_csv(path, nrows=sample, low_memory=False)
    # Parse the date-ish columns so they summarise as dates rather than strings.
    for col, fmt in (("issue_d", "%b-%y"), ("issue_date", "%B %Y")):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
    return df


# def normalise_missing(df: pd.DataFrame) -> pd.DataFrame:
#     """Turn placeholder strings ('', ' ', 'N/A', ...) into real NaNs."""
#     out = df.copy()
#     for col in out.columns:
#         if out[col].dtype == object:
#             stripped = out[col].astype("string").str.strip()
#             out[col] = stripped.mask(stripped.str.lower().isin(NULL_TOKENS))
#     return out


def infer_semantic_type(s: pd.Series) -> str:
    """Label a column beyond its raw dtype: id / binary / categorical / ..."""
    non_null = s.dropna()
    if non_null.empty:
        return "empty"

    n_unique = non_null.nunique()
    ratio = n_unique / len(non_null)

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"
    if pd.api.types.is_bool_dtype(s) or n_unique == 2:
        return "binary"
    if n_unique == 1:
        return "constant"
    if pd.api.types.is_numeric_dtype(s):
        is_whole = bool((non_null % 1 == 0).all())
        # Near-unique integers are almost always identifiers, not measurements.
        if ratio > 0.99 and is_whole:
            return "identifier (numeric)"
        if is_whole:
            if n_unique <= CATEGORICAL_MAX_UNIQUE:
                return "categorical (integer-coded)"
            return "numeric (discrete)"
        return "numeric (continuous)"
    if ratio > 0.99:
        return "identifier (text)"
    if n_unique <= CATEGORICAL_MAX_UNIQUE or ratio <= CATEGORICAL_MAX_RATIO:
        return "categorical"
    return "text / high-cardinality"


def type_table(df: pd.DataFrame) -> pd.DataFrame:
    """Section 1: dtype, inferred type, cardinality and an example value."""
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = s.dropna()
        rows.append(
            {
                "column": col,
                "dtype": str(s.dtype),
                "inferred_type": infer_semantic_type(s),
                "n_unique": int(non_null.nunique()),
                "unique_ratio": round(non_null.nunique() / len(s), 4) if len(s) else np.nan,
                "example": (str(non_null.iloc[0])[:40] if not non_null.empty else ""),
                "memory_mb": round(s.memory_usage(deep=True) / 1024**2, 2),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def missing_table(raw: pd.DataFrame, clean: pd.DataFrame) -> pd.DataFrame:
    """Section 2: nulls per column, split into true NaNs and placeholders."""
    n = len(clean)
    rows = []
    for col in clean.columns:
        native = int(raw[col].isna().sum())
        total = int(clean[col].isna().sum())
        rows.append(
            {
                "column": col,
                "n_missing": total,
                "pct_missing": round(100 * total / n, 2) if n else np.nan,
                "n_native_nan": native,
                "n_placeholder": total - native,  # '', 'N/A', 'unknown', ...
                "n_present": n - total,
                "complete": total == 0,
            }
        )
    return pd.DataFrame(rows).set_index("column").sort_values("pct_missing", ascending=False)


def summarise_numerical_column(s: pd.Series) -> None:
    """Print describe() plus skew/kurtosis for a numerical column."""
    non_null = s.dropna()
    desc = non_null.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    print(desc.to_string())
    print(f"  skew={non_null.skew():.3f}  kurtosis={non_null.kurtosis():.3f}  "
          f"zeros={int((non_null == 0).sum())}  negatives={int((non_null < 0).sum())}")


def summarise_categorical_column(s: pd.Series) -> None:
    """Print value counts and top categories for a categorical column, keeping NaN as its own category."""
    counts = s.value_counts(dropna=False)
    print(f"  n_unique={counts.size}")
    print("  top values:")
    top = counts.head(15)
    pct = (100 * top / len(s)).round(2)
    print(pd.DataFrame({"count": top, "pct_of_rows": pct}).to_string())
    if counts.size > 15:
        print(f"  ... {counts.size - 15} further categories")


def summarise_datetime_column(s: pd.Series) -> None:
    """Print describe() and a by-year breakdown for a datetime column."""
    non_null = s.dropna()
    print(non_null.describe().to_string())
    print("  by year:")
    print(non_null.dt.year.value_counts().sort_index().to_string())


def summarise_column(s: pd.Series, name: str) -> None:
    """Section 3: dispatch to the summary appropriate to the column's type."""
    kind = infer_semantic_type(s)
    print(f"\n--- {name}  [dtype={s.dtype}, inferred={kind}] ---")

    non_null = s.dropna()
    if non_null.empty:
        print("  (all values missing)")
        return

    if pd.api.types.is_numeric_dtype(s) and kind not in {"binary", "categorical (integer-coded)"}:
        summarise_numerical_column(s)
    elif pd.api.types.is_datetime64_any_dtype(s):
        summarise_datetime_column(s)
    else:
        summarise_categorical_column(s)


def main() -> None:
    print(f"Loading {DATA_PATH} ...")
    raw = load(DATA_PATH, SAMPLE)
    df = raw
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Fully duplicated rows: {int(df.duplicated().sum()):,}")

    print("\n" + "=" * 100)
    print("1. VARIABLE TYPES")
    print("=" * 100)
    print(type_table(df).to_string())

    print("\n" + "=" * 100)
    print("2. MISSING VALUE ANALYSIS")
    print("=" * 100)
    miss = missing_table(raw, df)
    print(miss.to_string())
    incomplete = miss[~miss["complete"]]
    print(f"\nColumns with missing data: {len(incomplete)} of {df.shape[1]}")
    print(f"Rows with at least one missing value: {int(df.isna().any(axis=1).sum()):,} "
          f"({100 * df.isna().any(axis=1).mean():.2f}%)")
    print(f"Complete rows: {int((~df.isna().any(axis=1)).sum()):,}")

    print("\n" + "=" * 100)
    print("3. PER-COLUMN SUMMARY")
    print("=" * 100)
    numerical_cols, datetime_cols, categorical_cols = [], [], []
    for col in df.columns:
        s = df[col]
        kind = infer_semantic_type(s)
        if pd.api.types.is_numeric_dtype(s) and kind not in {"binary", "categorical (integer-coded)"}:
            numerical_cols.append(col)
        elif pd.api.types.is_datetime64_any_dtype(s):
            datetime_cols.append(col)
        else:
            categorical_cols.append(col)

    for label, cols in (
        ("Numerical columns", numerical_cols),
        ("Datetime columns", datetime_cols),
        ("Categorical columns", categorical_cols),
    ):
        print(f"\n--- {label} ---")
        for col in cols:
            summarise_column(df[col], col)

    print("\n" + "=" * 100)
    print("Numeric columns, describe() side by side")
    print("=" * 100)
    num = df.select_dtypes(include=[np.number])
    if not num.empty:
        print(num.describe().T.to_string())


if __name__ == "__main__":
    main()

"""Object-oriented column-level analysis, reusable across datasets.

Replaces repeatedly copy-pasting `loan_column_analysis.py` into notebook
cells for each new CSV (`bureau.csv`, `credit_card_balance.csv`, ...) with
table objects that take no dataframe at construction time, so one instance
can be reused across many dataframes via `.plot_table(df)`.

Three broad table families, each possibly with further subclasses later
(e.g. a stricter/looser TypeTable variant, a MissingValueTable that also
splits out placeholder strings, per-semantic-type summary tables beyond
just numerical/categorical):

    1. TypeTable             - dtype, inferred semantic type, cardinality.
    2. MissingValueTable     - null counts/percentages per column.
    3. NumericalSummaryTable / CategoricalSummaryTable - per-column stats,
       split by inferred type (describe() vs. value_counts()).

`infer_semantic_type()` is kept as a public helper, not a method - used
both internally by the table classes and standalone by callers that need
to branch on a column's semantic type directly.

Requires the `data_science` conda environment (pandas, numpy):

    conda activate data_science
    python credit_risk/column_analysis.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/column_analysis.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# A column with few distinct values relative to its length is treated as
# categorical even when stored as text or as an integer-valued float.
CATEGORICAL_MAX_UNIQUE = 50
CATEGORICAL_MAX_RATIO = 0.05

def basic_data_summary(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    print(f"Fully duplicated rows: {int(df.duplicated().sum()):,}")

    return

def infer_semantic_type(s: pd.Series) -> str:
    """Label a column beyond its raw dtype: id / binary / categorical / ...

    Public helper - reused directly by callers that need to branch on a
    column's semantic type without going through one of the table classes
    below (e.g. to build a numerical_cols/categorical_cols split).
    """
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


class ColumnSummaryTable(ABC):
    """Common interface every column-analysis table object must satisfy.

    Subclasses hold only configuration/thresholds on `self` - the dataframe
    is passed into `plot_table()`, not the constructor, so one instance is
    reusable across many dataframes/datasets.
    """

    @abstractmethod
    def plot_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build and return this table's dataframe for `df`. Print-ready via
        `.to_string()` - does not print or plot anything itself."""
        raise NotImplementedError


class TypeTable(ColumnSummaryTable):
    """Section 1: dtype, inferred semantic type, cardinality, example value."""

    def plot_table(self, df: pd.DataFrame) -> pd.DataFrame:
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
                    "unique_ratio": round(non_null.nunique() / len(s), 4) if len(s) else float("nan"),
                    "example": (str(non_null.iloc[0])[:40] if not non_null.empty else ""),
                    "memory_mb": round(s.memory_usage(deep=True) / 1024**2, 2),
                }
            )
        return pd.DataFrame(rows).set_index("column")


class MissingValueTable(ColumnSummaryTable):
    """Section 2: null count/percentage per column, split into true NaNs and
    placeholder strings ('', 'N/A', 'unknown', ...) via `_normalise_missing`."""

    def _normalise_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn placeholder strings ('', ' ', 'N/A', ...) into real NaNs -
        the "clean" counterpart to the caller's raw `df`, so missingness
        hidden behind a placeholder string is counted too, not just native
        NaNs. Token set is local to this method, not shared module state."""
        null_tokens = {"", "na", "n/a", "nan", "null", "none", "-", "?", "unknown", "missing"}

        out = df.copy()
        for col in out.columns:
            if out[col].dtype == object:
                stripped = out[col].astype("string").str.strip()
                out[col] = stripped.mask(stripped.str.lower().isin(null_tokens))
        return out

    def plot_table(self, df: pd.DataFrame) -> pd.DataFrame:
        raw = df
        clean = self._normalise_missing(df)

        n = len(clean)
        rows = []
        for col in clean.columns:
            native = int(raw[col].isna().sum())
            total = int(clean[col].isna().sum())
            rows.append(
                {
                    "column": col,
                    "n_missing": total,
                    "pct_missing": round(100 * total / n, 2) if n else float("nan"),
                    "n_native_nan": native,
                    "n_placeholder": total - native,  # '', 'N/A', 'unknown', ...
                    "n_present": n - total,
                    "complete": total == 0,
                }
            )
        return pd.DataFrame(rows).set_index("column").sort_values("pct_missing", ascending=False)


class NumericalSummaryTable(ColumnSummaryTable):
    """Section 3a: describe()-style stats (min/25/50/75/max), one row per
    numerical column."""

    def _is_numerical(self, s: pd.Series) -> bool:
        """True for columns this table summarises - numeric dtype whose
        inferred semantic type isn't really categorical (binary /
        integer-coded), matching the original numerical_cols convention."""
        kind = infer_semantic_type(s)
        return pd.api.types.is_numeric_dtype(s) and kind not in {"binary", "categorical (integer-coded)"}

    def plot_table(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            s = df[col]
            if not self._is_numerical(s):
                continue

            non_null = s.dropna()
            # describe()'s default percentiles are already 25/50/75, plus
            # min/max always included - no need to pass percentiles explicitly.
            row = non_null.describe().to_dict()
            row["column"] = col
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")


class CategoricalSummaryTable(ColumnSummaryTable):
    """Section 3b: one row per categorical column, shaped to match
    `NumericalSummaryTable`: a `count`/`n_unique` pair followed by the top-N
    categories by frequency, flattened into `top_1` ... `top_N`."""

    TOP_N = 5

    def _is_categorical(self, s: pd.Series) -> bool:
        """True for columns this table summarises - everything
        `NumericalSummaryTable`/datetime columns don't cover, i.e. text
        columns plus the binary/integer-coded kinds that are numeric dtype
        but semantically categorical."""
        return not pd.api.types.is_datetime64_any_dtype(s) and (
            not pd.api.types.is_numeric_dtype(s)
            or infer_semantic_type(s) in {"binary", "categorical (integer-coded)"}
        )

    def plot_table(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in df.columns:
            s = df[col]
            if not self._is_categorical(s):
                continue

            # dropna=False keeps NaN as its own category, so missingness
            # shows up as a top category rather than being silently dropped.
            counts = s.value_counts(dropna=False)
            top = counts.head(self.TOP_N)

            row = {"column": col, "count": int(s.count()), "n_unique": int(counts.size)}
            for i, category in enumerate(top.index, start=1):
                row[f"top_{i}"] = category
            rows.append(row)
        return pd.DataFrame(rows).set_index("column")


def print_all_tables(df: pd.DataFrame) -> None:
    """Run all three table families against `df` and print each one.

    Covers the module's "Three broad table families" in one call - TypeTable,
    MissingValueTable, and the NumericalSummaryTable/CategoricalSummaryTable
    pair - rather than a caller instantiating and printing each separately.
    Prints only; returns nothing, since each table is print-ready via
    `.to_string()` and there's no single combined dataframe to hand back.
    """
    tables = (
        ("1. TYPE TABLE", TypeTable()),
        ("2. MISSING VALUE TABLE", MissingValueTable()),
        ("3a. NUMERICAL SUMMARY TABLE", NumericalSummaryTable()),
        ("3b. CATEGORICAL SUMMARY TABLE", CategoricalSummaryTable()),
    )
    for label, table in tables:
        print("\n" + "=" * 100)
        print(label)
        print("=" * 100)
        print(table.plot_table(df).to_string())


def main() -> None:
    """Smoke test: run every table against bureau.csv and print each one.

    Not a substitute for real unit tests - just confirms all four tables
    still run end-to-end against a real dataset after a change, the same
    way `loan_column_analysis.py`'s `main()` was used to sanity-check by eye.
    """
    data_path = Path(__file__).resolve().parent / "home_credit" / "bureau.csv"
    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    print_all_tables(df)

    # Every column should land in exactly one of the numerical/categorical
    # summary tables - catches a column silently falling through both
    # `_is_numerical`/`_is_categorical` (or being claimed by both).
    numerical_cols = set(NumericalSummaryTable().plot_table(df).index)
    categorical_cols = set(CategoricalSummaryTable().plot_table(df).index)
    assert not (numerical_cols & categorical_cols), "columns claimed by both summary tables"


if __name__ == "__main__":
    main()

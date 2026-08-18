# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Exploratory analysis of the [BigQuery Fintech Dataset](https://www.kaggle.com/datasets/mustafakeser4/bigquery-fintech-dataset) (`loan.csv` and `customer.csv` in `datasets/`, not committed to git — large CSVs). There is no build system, test suite, or package manifest — scripts are run directly with `python <script>.py` from the repo root, each writing its outputs (CSVs, PNGs) to disk as a side effect. Findings are written up in `documentation/main.tex` (LaTeX, compiled with `latexmk`).

## Running scripts

**Always run Python through the `data_science` conda environment — never the base/system Python.** Base Python is kept deliberately bare (pip only); all data-science packages (pandas, numpy, matplotlib, seaborn) live in `data_science`.

```
conda activate data_science
python credit_risk/loan_column_analysis.py
python credit_risk/clean_loan_dataset.py
python credit_risk/univariate_analysis.py
```

Without activating, call the environment's interpreter directly (no `python`/`conda` on PATH in this shell):

```
~/anaconda3/envs/data_science/python.exe credit_risk/loan_column_analysis.py
```

- `loan_column_analysis.py` — full column-level EDA of `datasets/loan.csv`: inferred semantic type per column (identifier/categorical/numeric/datetime/constant, beyond raw dtype), missing-value breakdown (native NaN vs. placeholder strings like `"n/a"`/`"unknown"`), and a `describe()`/`value_counts()` summary per column. Accepts `--sample N` to run on a row-capped subset for a faster pass; `--path`/`--output` to point at a different file.
- `clean_loan_dataset.py` — writes `datasets/loan_cleaned.csv` from `loan.csv`: drops identifier columns (`loan_id`, `customer_id` — kept in the raw file for a future join with `customer.csv`), duplicate columns (`funded_amount`, `description`, `issue_date`), and non-varying columns (`notes`, `pymnt_plan`), then normalises `type` to `lowercase_with_underscores` and folds casing-split levels (`Individual`/`INDIVIDUAL` → `individual`, `Joint App`/`JOINT` → `joint`). Every other column is asserted byte-identical to the source (`cleaned[col].equals(df[col])`) — the script only ever drops columns or rewrites `type`, so extending it to touch anything else means updating that assertion's exemption list too.
- `univariate_analysis.py` — `bar_chart(df, series, ...)` and `pie_chart(df, series, ...)` are generic, reusable single-column plotting functions (not tied to one dataset or column); `main()` currently runs them over `state` and `loan_status` from `loan_cleaned.csv` and saves PNGs to `documentation/figures/`.

There are no automated tests; validate changes by running the relevant script and inspecting the printed summary / saved PNGs (view them with the Read tool — don't assume from code alone).

## Conventions

- **Column dtype inference**: raw pandas dtypes are misleading on this dataset (e.g. `loan_id` is `int64` but is really an identifier, `issue_year` is `float64` but is really a small integer-coded category). `loan_column_analysis.infer_semantic_type()` is the canonical place this logic lives — reuse it rather than re-deriving ad hoc type checks in new scripts.
- **Missing-value convention**: nulls in this dataset can appear as real `NaN` or as placeholder strings (`""`, `"n/a"`, `"unknown"`, `"-"`, etc. — see `NULL_TOKENS` in `loan_column_analysis.py`). Always normalise placeholders before computing missingness; `normalise_missing()` does this.
- **Known data-quality issues in `loan.csv`** (documented in `documentation/main.tex`): `loan_amount`/`funded_amount` are identical on every row; `issue_d`/`issue_date` are the same date in two formats; `purpose`/`description` are the same field, but `description` has dirty casing/spacing (5,631 spurious levels vs. `purpose`'s clean 13); `pymnt_plan` is 99.99% `False`; `type` has a casing split (`Individual` vs `INDIVIDUAL`, `Joint App` vs `JOINT`). `clean_loan_dataset.py` is the fix for all of these except join-key columns, which are intentionally kept in the raw file.
- **Charts** (`univariate_analysis.py`) follow a fixed style: a single sequential blue hue for magnitude/count comparisons (one series, not identity), the validated 8-color categorical palette in fixed slot order for identity/part-to-whole, pie charts capped at `top_n=7` slices with the remainder folded into a gray "Other" wedge, bar-chart data labels auto-suppressed past 15 bars to avoid clutter. Keep new charts consistent with this rather than introducing a new palette or style per script.
- **`documentation/main.tex`** uses `\include{macros}` for a shared preamble (`macros.tex`, shared with sibling projects in this workspace) — don't duplicate package imports or custom commands already defined there. Build artifacts (`.aux`, `.log`, `.fdb_latexmk`, `.fls`, `.out`) are not committed; `main.pdf` is kept as the built output.

# Other Conventions 
- Scripts favor module-level constants (`DATA_PATH`, `SOURCE_PATH`, `OUTPUT_PATH`) over CLI argument parsing — follow this pattern rather than introducing argparse. Introduce these constants at the top of files.
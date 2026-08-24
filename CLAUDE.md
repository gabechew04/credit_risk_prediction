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
python credit_risk/bivariate_analysis.py
python credit_risk/correlations.py
```

Without activating, call the environment's interpreter directly (no `python`/`conda` on PATH in this shell):

```
~/anaconda3/envs/data_science/python.exe credit_risk/loan_column_analysis.py
```

Every script's `main()` calls seaborn/matplotlib plotting functions that end with `plt.show()`, which blocks on an interactive window until it's closed — expect to close each window in turn when running a script that produces multiple figures.

- `loan_column_analysis.py` — full column-level EDA of a CSV (`DATA_PATH`, currently `datasets/customer.csv`): inferred semantic type per column (identifier/categorical/numeric/datetime/constant/binary, beyond raw dtype, via `infer_semantic_type()`), a missing-value breakdown (`missing_table()`), and a per-column summary. The per-column summary buckets columns by inferred type first, then dispatches each to a type-specific summariser: `summarise_numerical_column()` (describe + skew/kurtosis), `summarise_datetime_column()` (describe + by-year counts), `summarise_categorical_column()` (value counts, keeping `NaN` as its own category via `dropna=False`).
- `clean_loan_dataset.py` — writes `datasets/loan_cleaned.csv` from `loan.csv`: drops identifier columns (`loan_id`, `customer_id` — kept in the raw file for a future join with `customer.csv`), duplicate columns (`funded_amount`, `description`, `issue_date`), and non-varying columns (`notes`, `pymnt_plan`), then normalises `type` to `lowercase_with_underscores` and folds casing-split levels (`Individual`/`INDIVIDUAL` → `individual`, `Joint App`/`JOINT` → `joint`). Every other column is asserted byte-identical to the source (`cleaned[col].equals(df[col])`) — the script only ever drops columns or rewrites `type`, so extending it to touch anything else means updating that assertion's exemption list too. Also derives `datasets/loan_status_recoded.csv`: drops `Current`/`In Grace Period` rows (no resolved outcome yet) and folds every non-`Fully Paid` status into `Missed Payment`, producing the binary target used throughout `univariate_analysis.py`/`bivariate_analysis.py`.
- `palette.py` — shared color constants (`CATEGORICAL_PALETTE`, `SEQUENTIAL_BLUE`, `OTHER_COLOR`, ink/gridline/surface tones); imported by the other analysis modules rather than redefined per file. See Conventions below for the actual values and usage.
- `univariate_analysis.py` — single-column plotting functions (`bar_chart`, `pie_chart`, `histogram_by_class`, `boxplot_by_class`) that take a dataframe plus column name(s) rather than a bare series and an explicit `ax`; each takes an optional `target_col` (binary-classification-aware, majority/minority split via `_binary_classes()`) that splits/hues the plot when given, or plots the whole column when `None`. Imports colors from `palette.py`. `main()` runs them over `loan_status_recoded.csv`.
- `bivariate_analysis.py` — two-column plotting functions, each taking an explicit `ax: plt.Axes` (no default — the caller always creates the figure/axes) and doing plotting only, no `plt.show()`: `numerical_and_numerical_scatter`, `categorical_and_numerical_scatter`, `categorical_and_numerical_boxplot`, `categorical_and_categorical_box` (missed-payment *rate* heatmap via `pd.crosstab(..., aggfunc="mean")`), `categorical_and_categorical_count` (raw-count heatmap via plain `pd.crosstab`). Imports `CATEGORICAL_PALETTE` from `palette.py` and `_binary_classes` from `univariate_analysis.py` so target-class colors match across both modules. `main()` builds the figure/axes and calls `plt.show()`/`savefig` itself — the plotting functions never do.
- `correlations.py` — `correlation_heatmap(df, ax, **kwargs)` computes `df.corr()` and draws it via `sns.heatmap`, setting only the title; creating `ax`, verifying `df`'s columns are numerical, and choosing a cmap are the caller's responsibility. `main()` selects the numerical columns from `loan_status_recoded.csv` and handles the rest.

## Figures

`documentation/figures/bivariate/` is split by plot kind, one subdirectory per pairing:
- `num_vs_num/` — numerical × numerical scatter plots.
- `num_vs_cat_scat/` — categorical × numerical scatter plots.
- `num_vs_cat_box/` — categorical × numerical boxplots.
- `cat_vs_cat_box/` — categorical × categorical, saved as a single figure with two side-by-side heatmaps (missed-payment rate on the left, raw counts on the right).

There are no automated tests; validate changes by running the relevant script and inspecting the printed summary / saved PNGs (view them with the Read tool — don't assume from code alone).

## Conventions

- **Constants over CLI args**: scripts favor module-level constants (`DATA_PATH`, `SOURCE_PATH`, `OUTPUT_PATH`, `SAMPLE`) at the top of the file over `argparse` — follow this pattern rather than introducing CLI parsing. To point a script at a different file or row-capped sample, edit the constant, don't add a flag.
- **Single-responsibility functions**: favor several small, single-purpose functions (e.g. `summarise_numerical_column`/`summarise_categorical_column`/`summarise_datetime_column`, or a plotting function that only plots vs. `main()` that owns figure creation/`show`/`savefig`) over one function handling multiple concerns — makes each piece independently checkable.
- **Optional-argument branching**: when a parameter (e.g. `target_col: str | None = None`) toggles behavior, don't gate the whole function body behind one `if target_col is not None:` at the top. Branch locally, at each spot the two cases actually differ (e.g. `bar_chart`/`histogram_by_class`/`boxplot_by_class` each branch separately at the plotting call, the title string, and the count-label step) and let the rest of the function run unconditionally. This keeps the shared logic shared instead of duplicated inside one large conditional block.
- **Column dtype inference**: raw pandas dtypes are misleading on this dataset (e.g. `loan_id` is `int64` but is really an identifier, `issue_year` is `float64` but is really a small integer-coded category). `loan_column_analysis.infer_semantic_type()` is the canonical place this logic lives — reuse it rather than re-deriving ad hoc type checks in new scripts.
- **Known data-quality issues in `loan.csv`** (documented in `documentation/main.tex`): `loan_amount`/`funded_amount` are identical on every row; `issue_d`/`issue_date` are the same date in two formats; `purpose`/`description` are the same field, but `description` has dirty casing/spacing (5,631 spurious levels vs. `purpose`'s clean 13); `pymnt_plan` is 99.99% `False`; `type` has a casing split (`Individual` vs `INDIVIDUAL`, `Joint App` vs `JOINT`). `clean_loan_dataset.py` is the fix for all of these except join-key columns, which are intentionally kept in the raw file.
- **Color palette** lives in `palette.py`, imported by `univariate_analysis.py`, `bivariate_analysis.py`, and (when needed) `correlations.py` rather than redefined per file:
  - `CATEGORICAL_PALETTE` — 8 fixed hex colors in slot order (`#2a78d6` blue, `#eb6834` orange, `#1baf7a` aqua, `#eda100` yellow, `#e87ba4` magenta, `#008300` green, `#4a3aa7` violet, `#e34948` red). For a binary majority/minority chart, build `{majority_class: CATEGORICAL_PALETTE[0], minority_class: CATEGORICAL_PALETTE[1]}` and pass it via the plotting function's `**kwargs` (e.g. `palette=` to `sns.histplot`/`sns.scatterplot`) — the slot order is fixed, not re-picked per chart.
  - `SEQUENTIAL_BLUE` (`#2a78d6`) — single-hue default for a magnitude/count comparison (one series, not an identity split).
  - `OTHER_COLOR` (`#c3c2b7`, muted gray) — for a folded "Other" category/wedge.
  - `INK_PRIMARY`/`INK_SECONDARY`/`INK_MUTED`/`GRIDLINE`/`SURFACE` — text/gridline/background tones for chart styling.
  - `bar_chart`'s per-bar count labels only render when a `palette` kwarg is supplied *and* the category count is within `MAX_LABELLED_BARS` (15), since the label logic matches drawn bars back to classes by facecolor.
- **Heatmap cmaps are deliberately left unset**: `categorical_and_categorical_box`/`categorical_and_categorical_count` in `bivariate_analysis.py` and `correlation_heatmap` in `correlations.py` never pass an explicit `cmap` to `sns.heatmap`, so all three render with seaborn's default heatmap colormap — keeping every heatmap in the project visually consistent. Don't hardcode a `cmap` on a new heatmap unless deliberately changing this project-wide.
- **`documentation/main.tex`** uses `\include{macros}` for a shared preamble (`macros.tex`, shared with sibling projects in this workspace) — don't duplicate package imports or custom commands already defined there. Build artifacts (`.aux`, `.log`, `.fdb_latexmk`, `.fls`, `.out`) are not committed; `main.pdf` is kept as the built output.

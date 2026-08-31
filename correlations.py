"""Correlation heatmap of the numerical columns in credit_risk/datasets/loan_status_recoded.csv.

Requires the `data_science` conda environment (pandas, matplotlib, seaborn):

    conda activate data_science
    python credit_risk/correlations.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/correlations.py
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency

DATA_DIR = Path(__file__).resolve().parent / "datasets"
FIG_DIR = Path(__file__).resolve().parent / "documentation" / "figures"
CORRELATIONS_FIG_DIR = FIG_DIR / "correlations"

NUM_COLS = ("loan_amount", "int_rate", "installment")
CAT_COLS = ("term", "state", "grade", "type", "purpose")


def correlation_heatmap(df: pd.DataFrame, ax: plt.Axes, **kwargs) -> None:
    """Heatmap of the pairwise correlation matrix of `df`'s columns.

    Only draws onto `ax` and sets its title - creating `ax`, checking that
    `df`'s columns are numerical, and choosing a color palette are the
    caller's responsibility (see `bivariate_analysis.py` for the same split
    of concerns).

    Parameters
    ----------
    df : dataframe of numerical columns to correlate.
    ax : Axes to plot onto.
    **kwargs : forwarded to `seaborn.heatmap` (e.g. `cmap`, `annot`).
    """
    corr = df.corr()
    sns.heatmap(corr, ax=ax, **kwargs)
    ax.set_title("Correlation heatmap")


def _cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Cramér's V association strength between two categorical columns.

    Bounded in [0, 1] (0 = no association, 1 = perfect association), unlike
    the raw chi-squared statistic - lets categorical pairs be compared on the
    same 0-1 scale as a numerical Pearson correlation.
    """
    contingency = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency)[0]
    n = contingency.sum().sum()
    r, k = contingency.shape
    return np.sqrt((chi2 / n) / (min(r - 1, k - 1)))


def cramers_v_heatmap(df: pd.DataFrame, ax: plt.Axes, **kwargs) -> None:
    """Heatmap of pairwise Cramér's V between `df`'s categorical columns.

    Only draws onto `ax` and sets its title - creating `ax`, checking that
    `df`'s columns are categorical, and choosing a color palette are the
    caller's responsibility (see `bivariate_analysis.py` for the same split
    of concerns).

    Parameters
    ----------
    df : dataframe of categorical columns to associate.
    ax : Axes to plot onto.
    **kwargs : forwarded to `seaborn.heatmap` (e.g. `cmap`, `annot`).
    """
    cols = df.columns
    matrix = pd.DataFrame(1.0, index=cols, columns=cols)
    for col1, col2 in combinations(cols, 2):
        v = _cramers_v(df[col1], df[col2])
        matrix.loc[col1, col2] = v
        matrix.loc[col2, col1] = v

    sns.heatmap(matrix, ax=ax, **kwargs)
    ax.set_title("Cramér's V heatmap")


def main() -> None:
    recoded_path = DATA_DIR / "loan_status_recoded.csv"
    print(f"Reading {recoded_path} ...")
    recoded = pd.read_csv(recoded_path, low_memory=False)

    CORRELATIONS_FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    correlation_heatmap(recoded[list(NUM_COLS)], ax=ax, annot=True, fmt=".2f")

    fig_path = CORRELATIONS_FIG_DIR / "numerical_correlation_heatmap.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()

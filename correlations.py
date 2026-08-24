"""Correlation heatmap of the numerical columns in credit_risk/datasets/loan_status_recoded.csv.

Requires the `data_science` conda environment (pandas, matplotlib, seaborn):

    conda activate data_science
    python credit_risk/correlations.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/correlations.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent / "datasets"
FIG_DIR = Path(__file__).resolve().parent / "documentation" / "figures"
CORRELATIONS_FIG_DIR = FIG_DIR / "correlations"

NUM_COLS = ("loan_amount", "int_rate", "installment")


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

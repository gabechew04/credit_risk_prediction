"""Decision-tree-based quantile bucketing for scorecard feature engineering.

Provides `TreeBucketer`, a scikit-learn transformer that buckets each
numerical column independently by fitting a shallow `DecisionTreeClassifier`
against the target and reading off the tree's split thresholds as bucket
edges - an alternative to plain quantile bucketing (`pd.qcut`) that lets the
bucket boundaries follow wherever the data actually separates goods from
bads, rather than an arbitrary equal-frequency split.

Requires the `data_science` conda environment (pandas, scikit-learn):

    conda activate data_science
    python credit_risk/tree_bucketer.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/tree_bucketer.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.tree import DecisionTreeClassifier

MIN_SAMPLES_LEAF = 0.05


class TreeBucketer(BaseEstimator, TransformerMixin):
    """Bucket each numerical column using a per-column decision tree's splits.

    For every input column, fits a `DecisionTreeClassifier` against `y` on
    that column alone, then reads the fitted tree's internal split thresholds
    as bucket edges - a column with no splits collapses to a single
    (-inf, inf] bucket. `min_samples_leaf` is a fraction of the training rows
    (see `DecisionTreeClassifier`), so it directly caps how small the
    smallest resulting bucket can be, the same role `top_n`/`duplicates` play
    for plain quantile bucketing elsewhere in this project.

    Only fits/transforms - choosing which columns are numerical, and any
    join back onto the rest of the dataframe, are the caller's responsibility
    (see `correlations.py`/`bivariate_analysis.py` for the same split of
    concerns).
    """

    def __init__(self, min_samples_leaf: float = MIN_SAMPLES_LEAF, random_state: int | None = 42):
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TreeBucketer":
        self.bin_edges_: dict[str, np.ndarray] = {}
        for col in X.columns:
            self.bin_edges_[col] = self._fit_column(X[col], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for col in X.columns:
            edges = self.bin_edges_[col]
            out[col] = pd.cut(X[col], bins=edges, include_lowest=True)
        return out

    def _fit_column(self, x: pd.Series, y: pd.Series) -> np.ndarray:
        """One column's bin edges: the fitted tree's split thresholds, plus
        -inf/+inf so a value outside the training range still falls into the
        first/last bucket instead of becoming NaN (matching the qcut-based
        bucketing this replaces)."""
        tree = DecisionTreeClassifier(
            min_samples_leaf=self.min_samples_leaf, random_state=self.random_state,
        )
        tree.fit(x.to_frame(), y)

        # threshold is -2 (TREE_UNDEFINED) on leaf nodes - only interior split
        # nodes (feature != -2) carry a real threshold.
        is_split_node = tree.tree_.feature != -2
        thresholds = np.sort(tree.tree_.threshold[is_split_node])

        return np.concatenate(([-np.inf], thresholds, [np.inf]))


def main() -> None:
    """Small smoke test against a synthetic column, run via `python tree_bucketer.py`."""
    rng = np.random.default_rng(42)
    n = 2000
    X = pd.DataFrame({"x": rng.normal(size=n)})
    y = pd.Series((X["x"] > 0.5).astype(int), name="y")

    bucketer = TreeBucketer(min_samples_leaf=MIN_SAMPLES_LEAF)
    bucketer.fit(X, y)
    bucketed = bucketer.transform(X)

    print("Learned bin edges:", bucketer.bin_edges_["x"])
    print(bucketed["x"].value_counts().sort_index())


if __name__ == "__main__":
    main()

"""Reusable scikit-learn-style transformers for the WOE scorecard pipeline.

Pulls the helper functions that were duplicated between the application-level
and bureau-level sections of `scorecard.ipynb` into one module, so both runs
of the pipeline call the same code rather than each keeping their own copy (a
fix applied to one copy previously wasn't reflected in the other - see the
`group_children_count` NameError this module fixes).

Every transformer here is a small `BaseEstimator`/`TransformerMixin` class
with a real `fit`/`transform`, same shape as `tree_bucketer.TreeBucketer` -
consistent syntax across the pipeline regardless of whether a given step
actually needs to learn anything from training data:
  - `PresenceBinarizer`, `ManualCategoryGrouper`: stateless column rewrites -
    what a bucket maps to doesn't depend on training data, so `fit()` is a
    no-op, but the class still exposes the same fit/transform interface as
    the stateful steps. Configuration (which columns, which mapping) is
    passed in at construction time by the caller - there is no module-level
    default, since which columns/groupings apply is specific to whichever
    dataset's pipeline is being built (application-level vs. bureau-level).
  - `WOEEncoder`: WOE values and the rare-category merge both have to be
    *learned* from training data and then applied unchanged to validation -
    `fit()` does real work here. Not yet implemented - see class docstring.

Requires the `data_science` conda environment (pandas, scikit-learn):

    conda activate data_science
    python credit_risk/scorecard_transformers.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/scorecard_transformers.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PresenceBinarizer(BaseEstimator, TransformerMixin):
    """Turn each of `cols` into a 0 (no events) / 1 (>=1 event) flag.

    Stateless - `cols` is fixed at construction time, not learned from
    training data - so `fit()` does nothing but is still provided to keep
    the same fit/transform interface as the stateful steps in this module.

    Parameters
    ----------
    cols : columns to binarize, e.g. ["NUM_LATE_30D", "NUM_DPD_30"] - counts
        of events collapsed to a simple presence flag rather than keeping
        every individual count as its own bucket.
    """

    def __init__(self, cols: list[str]):
        self.cols = cols

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "PresenceBinarizer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col in self.cols:
            out[col] = (out[col] > 0).astype(int)
        return out


class ManualCategoryGrouper(BaseEstimator, TransformerMixin):
    """Apply hand-specified category groupings, column by column.

    `group_maps` gives an explicit {old_label: new_label} dict per column -
    a value not present in a column's map (e.g. a category seen only in
    validation, never in training) keeps its own raw label rather than
    raising or becoming NaN, the same "no information" fallback used
    downstream by WOE encoding.

    CNT_CHILDREN is handled separately (`_group_children_count`) rather than
    through `group_maps`, since it collapses a numeric range to a label
    rather than remapping existing category strings - it doesn't fit
    `group_maps`'s dict-per-column shape. Stateless like `PresenceBinarizer`
    - the groupings are fixed at construction time, not learned.

    Parameters
    ----------
    group_maps : {column: {old_label: new_label}} - the specific groupings
        to apply, supplied by the caller (e.g. scorecard.ipynb's
        MANUAL_GROUP_MAPS) rather than defaulted here, since which columns
        and groupings apply is specific to whichever dataset's pipeline is
        being built.
    """

    def __init__(self, group_maps: dict[str, dict]):
        self.group_maps = group_maps

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ManualCategoryGrouper":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for col, mapping in self.group_maps.items():
            if col not in out.columns:
                continue
            out[col] = out[col].astype(str).map(mapping).fillna(out[col].astype(str))
        if "CNT_CHILDREN" in out.columns:
            out["CNT_CHILDREN"] = self._group_children_count(out["CNT_CHILDREN"])
        return out

    @staticmethod
    def _group_children_count(x: pd.Series) -> pd.Series:
        """CNT_CHILDREN: every value above 2 becomes the single label "2+"."""
        x = x.astype(str)
        return x.where(x.astype(float) <= 2, "2+")


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Weight-of-Evidence encoder, learned from training data only.

    For feature column x and target y: WOE_b = ln(%good_b / %bad_b) per
    bucket b of x, where good = y==0, bad = y==1, and the percentages are
    each bucket's share of all goods/bads (not of the bucket itself).
    `epsilon` is Laplace-style smoothing added to both counts so a bucket
    with zero goods or zero bads doesn't produce ln(0) / division by zero.

    Pure WOE encoding only - no rare-category merging. Bucketing/grouping
    (tree-based numerical bucketing, binarization, manual category grouping)
    is expected to already be done by the time `X` reaches this transformer;
    compose it after those steps rather than folding grouping into it.

    `transform(X)` maps every cell to its training-derived WOE value; a
    bucket present in `X` but never seen during `fit` (no WOE value learned
    for it) maps to 0 - i.e. "no information" - rather than raising or
    silently propagating NaN into a downstream model.
    """

    def __init__(self, epsilon: float = 0.5):
        self.epsilon = epsilon

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "WOEEncoder":
        self.woe_maps_: dict[str, pd.Series] = {
            col: self._compute_woe_map(X[col], y) for col in X.columns
        }
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                col: X[col].map(self.woe_maps_[col]).astype(object).fillna(0.0).astype(float)
                for col in X.columns
            },
            index=X.index,
        )

    def _compute_woe_map(self, x: pd.Series, y: pd.Series) -> pd.Series:
        table = pd.crosstab(x, y)
        goods = table.get(0, 0) + self.epsilon
        bads = table.get(1, 0) + self.epsilon
        pct_good = goods / goods.sum()
        pct_bad = bads / bads.sum()
        return np.log(pct_good / pct_bad)


def main() -> None:
    """Smoke test for all three transformers, run via
    `python scorecard_transformers.py`."""
    df = pd.DataFrame(
        {
            "NUM_LATE_30D": [0, 1, 3, 0],
            "CODE_GENDER": ["F", "M", "XNA", "F"],
            "CNT_CHILDREN": [0, 1, 5, 2],
            "OTHER_COL": ["a", "b", "c", "d"],
        }
    )
    y = pd.Series([0, 1, 1, 0])

    binarizer = PresenceBinarizer(cols=["NUM_LATE_30D"])
    print("Binarized:")
    print(binarizer.fit_transform(df))

    grouped = ManualCategoryGrouper(
        group_maps={"CODE_GENDER": {"F": "F", "M": "M", "XNA": "M"}},
    ).fit_transform(df)
    print("\nManually grouped:")
    print(grouped)

    # WOEEncoder expects grouping/bucketing already done - fit on the
    # already-grouped frame above, not the raw df.
    encoder = WOEEncoder()
    encoder.fit(grouped, y)
    print("\nWOE maps:")
    print(encoder.woe_maps_)
    print("\nWOE-encoded:")
    print(encoder.transform(grouped))


if __name__ == "__main__":
    main()

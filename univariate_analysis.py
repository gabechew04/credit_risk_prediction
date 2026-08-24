"""Univariate categorical analysis charts.

Four reusable chart functions, all keyed off a binary `target_col` (this is a
binary classification project):
  - bar_chart : horizontal STACKED bar of `cat_col` counts, sorted by total
                descending, split into the two `target_col` classes - the
                majority class as the bottom (first) segment, the minority
                class stacked on top, so its share is easy to spot per
                category. Capped at `top_n` categories, if given.
  - pie_chart : two side-by-side part-to-whole pies of `cat_col`, one per
                `target_col` class - each pie's slices sum to 100% of *that
                class's* rows, not the whole dataframe. Both panels share the
                same top-`top_n` category set and category -> color mapping
                (default 7) so they're directly comparable; the remainder
                folds into a single muted "Other" wedge per panel.
  - histogram_by_class : histogram of a numerical column, faceted into one
                panel per class of a target column (seaborn FacetGrid).
  - boxplot_by_class : boxplot of a numerical column, one box per class of a
                target column, on a shared axis (seaborn boxplot).

All four take the dataframe and column names as strings (`cat_col`/`num_col`,
`target_col`) rather than pre-sliced Series - columns are looked up from `df`
and a `KeyError` is raised if a name isn't found. `histogram_by_class`/
`boxplot_by_class` also accept `**kwargs` forwarded to the underlying seaborn
call.

Colors are the validated reference categorical/sequential palette from the
dataviz skill (light-mode slots) - not re-picked per chart.

Requires the `data_science` conda environment (pandas, matplotlib):

    conda activate data_science
    python credit_risk/univariate_analysis.py

Without activating, call the environment's interpreter directly:

    ~/anaconda3/envs/data_science/python.exe credit_risk/univariate_analysis.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import pandas as pd
import seaborn as sns

from palette import (
    CATEGORICAL_PALETTE,
    GRIDLINE,
    INK_MUTED,
    INK_PRIMARY,
    INK_SECONDARY,
    OTHER_COLOR,
    SEQUENTIAL_BLUE,
    SURFACE,
)

save_path = ""

DATA_DIR = Path(__file__).resolve().parent / "datasets" 
FIG_DIR = Path(__file__).resolve().parent / "documentation" / "master" / "figures"
UNIVARIATE_FIG_DIR = FIG_DIR / "univariate"
NUMERICAL_FIG_DIR = UNIVARIATE_FIG_DIR / "numerical"
CATEGORICAL_FIG_DIR = UNIVARIATE_FIG_DIR / "categorical"

# Above this many bars, per-bar count/pct labels clutter more than they
# inform; the axis carries the value instead.
MAX_LABELLED_BARS = 15


def _require_columns(df: pd.DataFrame, *cols: str) -> None:
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"'{col}' is not a column in df")


def _binary_classes(df: pd.DataFrame, target_col: str) -> tuple:
    """(majority_class, minority_class) of `target_col`, by row count.

    Raises if `target_col` doesn't have exactly two classes - the charts in
    this module assume a binary classification target.
    """
    class_counts = df[target_col].value_counts()
    if len(class_counts) != 2:
        raise ValueError(
            f"expected a binary `target_col`; '{target_col}' has {len(class_counts)} classes: "
            f"{list(class_counts.index)}"
        )
    return class_counts.index[0], class_counts.index[1]  # value_counts sorts descending

    

def bar_chart(
    df: pd.DataFrame,
    cat_col: str,
    ax: plt.Axes,
    target_col: str | None = None,
    top_n: int | None = None,
    **kwargs,
) -> None:
    """Horizontal bar chart of `cat_col` counts, optionally split by `target_col`.

    Only draws onto `ax` and sets its title/axis labels - palette, legend
    placement, `plt.savefig`, and `plt.show` are the caller's responsibility
    (see `bivariate_analysis.py` for the same split of concerns).

    Bars (one per `cat_col` category) are sorted by total count descending.
    If `target_col` is given (must be binary), each category gets two
    side-by-side bars via `seaborn.histplot(..., multiple="dodge")`, one per
    class; if `None`, a single bar per category is drawn.

    Parameters
    ----------
    df : the dataframe containing `cat_col` (and `target_col`, if given).
    cat_col : name of the categorical column to plot, one bar per value.
    ax : Axes to plot onto.
    target_col : name of a binary column to split each bar by, if given.
    top_n : keep only the `top_n` most frequent `cat_col` categories, if given.
    **kwargs : forwarded to `seaborn.histplot` (e.g. `palette`, `shrink`).
    """
    _require_columns(df, cat_col)
    if target_col is not None:
        _require_columns(df, target_col)
        majority_class, minority_class = _binary_classes(df, target_col)

    totals = df[cat_col].value_counts()
    if top_n is not None:
        totals = totals.head(top_n)
    # seaborn's categorical y-axis renders the first entry at the top (opposite
    # of raw matplotlib's barh), so descending here puts the largest on top.
    order = totals.sort_values(ascending=False).index.tolist()

    # An explicit ordered Categorical forces histplot's y-axis to use exactly
    # this category order (largest total at the top) - passing a plain
    # object-dtype column left seaborn to pick its own (mismatched) order.
    plot_df = df[df[cat_col].isin(order)].copy()
    plot_df[cat_col] = pd.Categorical(plot_df[cat_col], categories=order, ordered=True)

    # Narrower bars (shrink < 1) open a visible gap between categories, and an
    # edge color/width keeps same-color bars in neighboring categories from
    # blending into one another.
    kwargs.setdefault("shrink", 0.7)
    kwargs.setdefault("edgecolor", "white")
    kwargs.setdefault("linewidth", 1)

    if target_col is not None:
        sns.histplot(
            data=plot_df, y=cat_col, hue=target_col, hue_order=[majority_class, minority_class],
            multiple="dodge", ax=ax, **kwargs,
        )
    else:
        sns.histplot(data=plot_df, y=cat_col, ax=ax, **kwargs)

    title = f"{cat_col}: counts" + (f" by {target_col}" if target_col is not None else "")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel(cat_col)

    palette = kwargs.get("palette")
    if target_col is not None and palette is not None and len(order) <= MAX_LABELLED_BARS:
        class_counts = pd.crosstab(df[cat_col], df[target_col]).loc[order]

        # Read each class's actual drawn bar geometry rather than assuming a
        # dodge order from hue_order - not part of seaborn's public contract.
        # histplot doesn't label containers under multiple="dodge" (get_label()
        # returns the matplotlib default "_containerN"), so match by facecolor.
        def _container_for(cls):
            target_rgb = to_rgb(palette[cls])
            return next(c for c in ax.containers if to_rgb(c.patches[0].get_facecolor()) == target_rgb)

        majority_patches = dict(zip(order, _container_for(majority_class).patches))
        minority_patches = dict(zip(order, _container_for(minority_class).patches))

        x_max = totals.max()
        for cat in order:
            maj_patch, minr_patch = majority_patches[cat], minority_patches[cat]
            maj, minr = class_counts.loc[cat, majority_class], class_counts.loc[cat, minority_class]
            for patch, count in ((maj_patch, maj), (minr_patch, minr)):
                ax.text(
                    patch.get_width() + x_max * 0.01, patch.get_y() + patch.get_height() / 2,
                    f"{count:,}", va="center", ha="left", fontsize=8,
                )
        ax.set_xlim(0, x_max * 1.2)


def pie_chart(
    df: pd.DataFrame,
    cat_col: str,
    ax: plt.Axes,
    top_n: int | None = None,
    **kwargs,
) -> None:
    """Part-to-whole pie of `cat_col` for a single dataframe.

    Only draws onto `ax` and sets its title - filtering by class, colors,
    legend placement, `plt.savefig`, and `plt.show` are the caller's
    responsibility (see `bivariate_analysis.py` for the same split of
    concerns).

    Parameters
    ----------
    df : the (already filtered, if applicable) dataframe to plot.
    cat_col : name of the categorical column to plot, one wedge per value.
    ax : Axes to plot onto.
    top_n : keep only the `top_n` most frequent categories, folding the rest
        into a single "Other" wedge, if given.
    **kwargs : forwarded to `ax.pie` (e.g. `colors`, `autopct`, `startangle`).
    """
    _require_columns(df, cat_col)
    counts = df[cat_col].value_counts()
    if top_n is not None:
        head = counts.head(top_n)
        other_total = counts.iloc[top_n:].sum()
        if other_total > 0:
            head = pd.concat([head, pd.Series({"Other": other_total})])
        counts = head
    ax.pie(counts.values, **kwargs)
    ax.set_title(cat_col)
    ax.set_aspect("equal")


def _skew_kurt_label(series: pd.Series) -> str:
    """`skew=.. kurt=..` label text for one distribution.

    Kurtosis is pandas' excess kurtosis (0 = normal), matching `.skew()`'s
    Fisher-Pearson convention - the two are directly comparable at a glance.
    """
    return f"skew={series.skew():.2f} kurt={series.kurtosis():.2f}"


def histogram_by_class(
    df: pd.DataFrame,
    num_col: str,
    ax: plt.Axes,
    target_col: str | None = None,
    **kwargs,
) -> None:
    """Histogram of `num_col`, optionally split by `target_col`.

    Only draws onto `ax` and sets its title/axis labels - legend placement,
    `plt.savefig`, and `plt.show` are the caller's responsibility (see
    `bivariate_analysis.py` for the same split of concerns).

    Parameters
    ----------
    df : the dataframe containing `num_col` (and `target_col`, if given).
    num_col : name of the numerical column to histogram.
    ax : Axes to plot onto.
    target_col : name of a column to split the histogram by (`hue`), if
        given; if `None`, a single histogram of the whole column is drawn.
    **kwargs : forwarded to `seaborn.histplot` (e.g. `bins`, `stat`, `palette`).
    """
    _require_columns(df, num_col)
    if target_col is not None:
        _require_columns(df, target_col)

    # "step" draws only the outer outline of the distribution, not a border
    # around every individual bin.
    kwargs.setdefault("element", "step")

    sns.histplot(data=df, x=num_col, hue=target_col, ax=ax, **kwargs)
    title = f"{num_col} distribution" + (f" by {target_col}" if target_col is not None else "")
    ax.set_title(title)
    ax.set_xlabel(num_col)
    ax.set_ylabel("Count")

    if target_col is None:
        ax.text(
            0.97, 0.95, _skew_kurt_label(df[num_col].dropna()), transform=ax.transAxes,
            ha="right", va="top", fontsize=8,
        )
    else:
        classes = sorted(df[target_col].dropna().unique().tolist(), key=str)
        for i, cls in enumerate(classes):
            class_values = df.loc[df[target_col] == cls, num_col].dropna()
            ax.text(
                0.97, 0.95 - i * 0.08, f"{cls}: {_skew_kurt_label(class_values)}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
            )


def boxplot_by_class(
    df: pd.DataFrame,
    num_col: str,
    ax: plt.Axes,
    target_col: str | None = None,
    **kwargs,
) -> None:
    """Boxplot of `num_col`, optionally one box per `target_col` class.

    Only draws onto `ax` and sets its title/axis labels - legend placement,
    `plt.savefig`, and `plt.show` are the caller's responsibility (see
    `bivariate_analysis.py` for the same split of concerns).

    Parameters
    ----------
    df : the dataframe containing `num_col` (and `target_col`, if given).
    num_col : name of the numerical column to plot.
    ax : Axes to plot onto.
    target_col : name of a column to split into one box per level, if given;
        if `None`, a single box of the whole column is drawn.
    **kwargs : forwarded to `seaborn.boxplot` (e.g. `showfliers`, `whis`, `palette`).
    """
    _require_columns(df, num_col)

    data_range = df[num_col].max() - df[num_col].min()
    label_gap = 0.02 * data_range  # clears the topmost outlier marker

    if target_col is not None:
        _require_columns(df, target_col)
        classes = sorted(df[target_col].dropna().unique().tolist(), key=str)
        kwargs.setdefault("order", classes)
        sns.boxplot(data=df, x=target_col, y=num_col, ax=ax, **kwargs)

        order = kwargs["order"]
        class_maxes = []
        for i, cls in enumerate(order):
            class_values = df.loc[df[target_col] == cls, num_col].dropna()
            class_max = class_values.max()
            class_maxes.append(class_max)
            ax.text(
                i, class_max + label_gap, _skew_kurt_label(class_values),
                ha="center", va="bottom", fontsize=8,
            )
        ax.set_ylim(top=max(class_maxes) + 0.1 * data_range)
    else:
        sns.boxplot(data=df, y=num_col, ax=ax, **kwargs)

        values = df[num_col].dropna()
        ax.text(
            0, values.max() + label_gap, _skew_kurt_label(values),
            ha="center", va="bottom", fontsize=8,
        )
        ax.set_ylim(top=values.max() + 0.1 * data_range)

    title = num_col + (f" by {target_col}" if target_col is not None else "")
    ax.set_title(title)
    if target_col is not None:
        ax.set_xlabel(target_col)
    ax.set_ylabel(num_col)


def main() -> None:
    recoded_path = DATA_DIR / "master.csv"
    print(f"Reading {recoded_path} ...")
    recoded = pd.read_csv(recoded_path, low_memory=False).drop(labels=["customer_id"], axis=1)
    #target_col = "loan_status"

    CATEGORICAL_FIG_DIR.mkdir(parents=True, exist_ok=True)

    #majority_class, minority_class = _binary_classes(recoded, target_col)
    #palette = {majority_class: CATEGORICAL_PALETTE[0], minority_class: CATEGORICAL_PALETTE[1]}

    for cat_col in ("emp_length", "home_ownership", "verification_status", "state", "grade"):
        #print(f"\n`{cat_col}` x `{target_col}`:")
        #print(pd.crosstab(recoded[cat_col], recoded[target_col]).to_string())

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_chart(recoded, cat_col, ax=ax, top_n=7)
        # ax.legend(
        #     handles=[
        #         plt.Rectangle((0, 0), 1, 1, color=palette[majority_class]),
        #         plt.Rectangle((0, 0), 1, 1, color=palette[minority_class]),
        #     ],
        #     labels=[f"{majority_class} (majority)", f"{minority_class} (minority)"],
        #     loc="lower right",
        # )
        bar_path = CATEGORICAL_FIG_DIR / f"{cat_col}_bar_chart.png"
        fig.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(fig)
        print(f"Wrote {bar_path}")

        pie_fig, pie_ax = plt.subplots(1, 1, figsize=(14, 7))
        # for pie_ax, cls in zip(pie_axes, (majority_class, minority_class)):
        #     class_df = recoded[recoded[target_col] == cls]
        #     pie_chart(class_df, cat_col, ax=pie_ax, autopct=lambda p: f"{p:.1f}%" if p >= 3 else "")
        #     #pie_ax.set_title(f"{target_col} = {cls}")

        pie_chart(recoded, cat_col, ax=pie_ax, autopct=lambda p: f"{p:.1f}%" if p >= 3 else "", top_n=7)
        pie_ax.legend(
            pie_ax.patches, recoded[cat_col].value_counts().index,
            loc="center left", bbox_to_anchor=(1.0, 0.5),
        )
        pie_path = CATEGORICAL_FIG_DIR / f"{cat_col}_pie_chart.png"
        pie_fig.savefig(pie_path, dpi=150, bbox_inches="tight")
        plt.show()
        plt.close(pie_fig)
        print(f"Wrote {pie_path}")

    # NUMERICAL_FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    # for num_col in ("loan_amount", "int_rate", "installment", "annual_inc", "avg_cur_bal", "Tot_cur_bal"):
    #     fig, (hist_ax, box_ax) = plt.subplots(1, 2, figsize=(14, 5))
    #     histogram_by_class(recoded, num_col, ax=hist_ax)
    #     boxplot_by_class(recoded, num_col, ax=box_ax)

    #     num_path = NUMERICAL_FIG_DIR / f"{num_col}_hist_box.png"
    #     fig.savefig(num_path, dpi=150, bbox_inches="tight")
    #     plt.show()
    #     plt.close(fig)
    #     print(f"Wrote {num_path}")


if __name__ == "__main__":
    main()

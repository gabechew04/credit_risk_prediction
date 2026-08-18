"""Univariate categorical analysis charts.

Four reusable chart functions:
  - bar_chart : horizontal bar of category counts, sorted descending.
                One hue for every bar - this is a magnitude comparison of a
                single series, not an identity encoding, so no per-bar color.
  - pie_chart : part-to-whole pie, capped at `top_n` categories (default 7)
                with the remainder folded into a single "Other" wedge. Pies
                blur past ~7 segments, so the cap is enforced rather than
                left to the caller.
  - histogram_by_class : histogram of a numerical column, faceted into one
                panel per class of a target column (seaborn FacetGrid).
  - boxplot_by_class : boxplot of a numerical column, one box per class of a
                target column, on a shared axis (seaborn boxplot).

`bar_chart`/`pie_chart` take the parent dataframe (used only for `len(df)`, to
compute each category's share of all rows) and the column as a Series.
`histogram_by_class`/`boxplot_by_class` take the dataframe and the numerical/
target column names as strings, plus `**kwargs` forwarded to the underlying
seaborn call.

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
import pandas as pd
import seaborn as sns

DATA_DIR = Path(__file__).resolve().parent / "datasets"
FIG_DIR = Path(__file__).resolve().parent / "documentation" / "figures"

# Reference categorical palette, light mode, fixed slot order (dataviz skill).
CATEGORICAL_PALETTE = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]
OTHER_COLOR = "#c3c2b7"  # muted gray for the folded "Other" wedge
SEQUENTIAL_BLUE = "#2a78d6"  # single-hue default for a magnitude comparison

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Above this many bars, per-bar count/pct labels clutter more than they
# inform; the axis carries the value instead.
MAX_LABELLED_BARS = 15


def bar_chart(
    df: pd.DataFrame,
    series: pd.Series,
    top_n: int | None = None,
    save_path: Path | str | None = None,
    ax: plt.Axes | None = None,
):
    """Horizontal bar chart of `series` category counts, sorted descending.

    Parameters
    ----------
    df : the dataframe `series` was drawn from (used for `len(df)`, to
        express each bar's count as a % of all rows).
    series : the categorical column to plot.
    top_n : keep only the `top_n` most frequent categories, if given.
    save_path : if given, the figure is saved here (dpi=150) and closed.
    ax : plot onto an existing Axes instead of creating a new figure.
    """
    counts = series.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    counts = counts.sort_values(ascending=True)  # largest ends up at the top of barh

    pct = 100 * counts / len(df)

    created_fig = ax is None
    if created_fig:
        fig_height = max(3, 0.32 * len(counts) + 1)
        fig, ax = plt.subplots(figsize=(8, fig_height), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    bars = ax.barh(
        counts.index.astype(str), counts.values, color=SEQUENTIAL_BLUE, height=0.65, zorder=3
    )

    ax.set_xlabel("Count", color=INK_SECONDARY, fontsize=10)
    ax.set_title(f"{series.name}: category counts", color=INK_PRIMARY, fontsize=12, loc="left", pad=12)

    ax.grid(axis="x", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRIDLINE)
    ax.tick_params(axis="both", colors=INK_MUTED, labelsize=9, length=0)

    x_max = counts.values.max()
    if len(counts) <= MAX_LABELLED_BARS:
        for bar, count, p in zip(bars, counts.values, pct.values):
            ax.text(
                bar.get_width() + x_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({p:.1f}%)",
                va="center", ha="left", fontsize=8, color=INK_SECONDARY,
            )
        ax.set_xlim(0, x_max * 1.2)
    else:
        ax.set_xlim(0, x_max * 1.05)

    if created_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
        return fig, ax
    return ax


def pie_chart(
    df: pd.DataFrame,
    series: pd.Series,
    top_n: int = 7,
    save_path: Path | str | None = None,
    ax: plt.Axes | None = None,
):
    """Part-to-whole pie chart of `series` category shares.

    Capped at the `top_n` most frequent categories (categorical palette,
    fixed slot order); anything past that folds into one muted "Other"
    wedge, since pie slices blur once there are more than a handful.

    Parameters mirror `bar_chart`.
    """
    counts = series.value_counts(dropna=False)
    if len(counts) > top_n:
        head = counts.head(top_n)
        other_total = counts.iloc[top_n:].sum()
        counts = pd.concat([head, pd.Series({"Other": other_total})])

    pct = 100 * counts / len(df)

    n_slices = len(counts) - (1 if "Other" in counts.index else 0)
    colors = CATEGORICAL_PALETTE[:n_slices]
    if "Other" in counts.index:
        colors = colors + [OTHER_COLOR]

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(7, 7), facecolor=SURFACE)

    wedges, _, _ = ax.pie(
        counts.values,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops={"linewidth": 2, "edgecolor": SURFACE},
        autopct=lambda p: f"{p:.1f}%" if p >= 3 else "",
        pctdistance=0.75,
        textprops={"color": INK_PRIMARY, "fontsize": 9},
    )

    ax.set_title(f"{series.name}: category share", color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.legend(
        wedges,
        [f"{idx} ({p:.1f}%)" for idx, p in zip(counts.index.astype(str), pct.values)],
        loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )
    ax.set_aspect("equal")

    if created_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
        return fig, ax
    return ax


def _skew_kurt_label(series: pd.Series) -> str:
    """`skew=.. / kurt=..` label text for one class's distribution.

    Kurtosis is pandas' excess kurtosis (0 = normal), matching `.skew()`'s
    Fisher-Pearson convention - the two are directly comparable at a glance.
    """
    return f"skew={series.skew():.2f}\nkurt={series.kurtosis():.2f}"


def histogram_by_class(
    df: pd.DataFrame,
    num_col: str,
    target_col: str,
    save_path: Path | str | None = None,
    **kwargs,
):
    """Histogram of `num_col`, faceted into one panel per `target_col` class.

    Histograms overlaid on a single axis are unreadable past ~2 classes, so
    each class gets its own panel (seaborn `FacetGrid`), sharing x/y scales
    for comparability. Panels are colored by class, in the same fixed
    categorical order as `pie_chart`. Each panel is annotated with that
    class's skewness and (excess) kurtosis.

    Parameters
    ----------
    df : the dataframe containing both columns.
    num_col : name of the numerical column to histogram.
    target_col : name of the target/class column to facet by.
    save_path : if given, the figure is saved here (dpi=150) and closed.
    **kwargs : forwarded to `seaborn.histplot` for each facet (e.g. `bins`,
        `stat`, `log_scale`).
    """
    classes = sorted(df[target_col].dropna().unique().tolist(), key=str)
    palette = {cls: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, cls in enumerate(classes)}

    kwargs.setdefault("edgecolor", SURFACE)

    grid = sns.FacetGrid(
        df, col=target_col, hue=target_col, palette=palette, col_order=classes, hue_order=classes,
        col_wrap=4 if len(classes) > 4 else None, sharex=True, sharey=True, height=3.2, aspect=1.2,
    )
    grid.map_dataframe(sns.histplot, x=num_col, **kwargs)
    grid.set_titles(col_template="{col_name}")
    grid.set_axis_labels(num_col, "Count")
    grid.figure.suptitle(
        f"{num_col} distribution by {target_col}", x=0.02, ha="left", y=1.02, color=INK_PRIMARY, fontsize=12
    )
    grid.figure.patch.set_facecolor(SURFACE)

    for cls, ax in grid.axes_dict.items():
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRIDLINE)
        ax.tick_params(colors=INK_MUTED, labelsize=9)

        class_values = df.loc[df[target_col] == cls, num_col].dropna()
        ax.text(
            0.97, 0.95, _skew_kurt_label(class_values), transform=ax.transAxes,
            ha="right", va="top", fontsize=8, color=INK_SECONDARY,
        )

    if save_path is not None:
        grid.figure.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
        plt.close(grid.figure)
    return grid


def boxplot_by_class(
    df: pd.DataFrame,
    num_col: str,
    target_col: str,
    save_path: Path | str | None = None,
    ax: plt.Axes | None = None,
    **kwargs,
):
    """Boxplot of `num_col`, one box per `target_col` class on a shared axis.

    Boxes are colored by class, in the same fixed categorical order as
    `pie_chart`/`histogram_by_class`. Each box is annotated above its maximum
    with that class's skewness and (excess) kurtosis.

    Parameters
    ----------
    df : the dataframe containing both columns.
    num_col : name of the numerical column to plot.
    target_col : name of the target/class column, one box per level.
    save_path : if given, the figure is saved here (dpi=150) and closed.
    ax : plot onto an existing Axes instead of creating a new figure.
    **kwargs : forwarded to `seaborn.boxplot` (e.g. `showfliers`, `whis`, `order`).
    """
    classes = sorted(df[target_col].dropna().unique().tolist(), key=str)
    palette = {cls: CATEGORICAL_PALETTE[i % len(CATEGORICAL_PALETTE)] for i, cls in enumerate(classes)}

    kwargs.setdefault("order", classes)
    kwargs.setdefault("hue", target_col)
    kwargs.setdefault("palette", palette)
    kwargs.setdefault("legend", False)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=(max(4, 1.3 * len(classes)), 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    sns.boxplot(data=df, x=target_col, y=num_col, ax=ax, **kwargs)

    order = kwargs["order"]
    data_range = df[num_col].max() - df[num_col].min()
    label_gap = 0.02 * data_range  # clears the topmost outlier marker
    class_maxes = []
    for i, cls in enumerate(order):
        class_values = df.loc[df[target_col] == cls, num_col].dropna()
        class_max = class_values.max()
        class_maxes.append(class_max)
        ax.text(
            i, class_max + label_gap, _skew_kurt_label(class_values),
            ha="center", va="bottom", fontsize=8, color=INK_SECONDARY,
        )
    # Headroom for the two-line labels above the tallest box/whisker/outlier.
    ax.set_ylim(top=max(class_maxes) + 0.14 * data_range)

    ax.set_title(f"{num_col} by {target_col}", color=INK_PRIMARY, fontsize=12, loc="left", pad=12)
    ax.set_xlabel(target_col, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(num_col, color=INK_SECONDARY, fontsize=10)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRIDLINE)
    ax.tick_params(colors=INK_MUTED, labelsize=9)

    if created_fig:
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
            plt.close(fig)
        return fig, ax
    return ax


def main() -> None:
    path = DATA_DIR / "loan_cleaned.csv"
    print(f"Reading {path} ...")
    df = pd.read_csv(path, low_memory=False)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    for col in ("state", "loan_status"):
        series = df[col]
        print(f"\n`{col}`: {series.nunique()} unique values, {series.isna().sum()} missing")
        print(series.value_counts().to_string())

        bar_path = FIG_DIR / f"{col}_bar_chart.png"
        pie_path = FIG_DIR / f"{col}_pie_chart.png"

        bar_chart(df, series, save_path=bar_path)
        pie_chart(df, series, save_path=pie_path)

        print(f"\nWrote {bar_path}")
        print(f"Wrote {pie_path}")

    recoded_path = DATA_DIR / "loan_status_recoded.csv"
    print(f"\nReading {recoded_path} ...")
    recoded = pd.read_csv(recoded_path, low_memory=False)
    target_col = "loan_status"

    for num_col in ("loan_amount", "int_rate", "installment"):
        hist_path = FIG_DIR / f"{num_col}_by_{target_col}_hist.png"
        box_path = FIG_DIR / f"{num_col}_by_{target_col}_box.png"

        histogram_by_class(recoded, num_col, target_col, save_path=hist_path)
        boxplot_by_class(recoded, num_col, target_col, save_path=box_path)

        print(f"Wrote {hist_path}")
        print(f"Wrote {box_path}")


if __name__ == "__main__":
    main()

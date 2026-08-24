from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from palette import CATEGORICAL_PALETTE
from univariate_analysis import _binary_classes

DATA_DIR = Path(__file__).resolve().parent / "datasets"
FIG_DIR = Path(__file__).resolve().parent / "documentation" / "figures"
BIVARIATE_FIG_DIR = FIG_DIR / "bivariate"
NUM_VS_NUM_DIR = BIVARIATE_FIG_DIR / "num_vs_num"
NUM_VS_CAT_SCAT_DIR = BIVARIATE_FIG_DIR / "num_vs_cat_scat"
NUM_VS_CAT_BOX_DIR = BIVARIATE_FIG_DIR / "num_vs_cat_box"
CAT_VS_CAT_BOX_DIR = BIVARIATE_FIG_DIR / "cat_vs_cat_box"

class AxNoneError(Exception):
    pass

def numerical_and_numerical_scatter(
    df: pd.DataFrame,
    num_col1: str,
    num_col2: str,
    target: str,
    ax: plt.Axes,
    **kwargs
) -> None:


    sns.scatterplot(data = df, x = num_col1, y = num_col2, hue = target, ax=ax, **kwargs)
    ax.set_title(f"Scatterplot of {num_col1} against {num_col2}, grouped by {target}")
    ax.set_xlabel(num_col1)
    ax.set_ylabel(num_col2)
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()


def categorical_and_numerical_scatter(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    target: str,
    ax: plt.Axes,
    **kwargs
) -> None:


    sns.stripplot(data=df, x=cat_col, y=num_col, hue=target, ax=ax, **kwargs)
    ax.set_title(f"Scatterplot of {cat_col} against {num_col}, grouped by {target}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()


def categorical_and_numerical_boxplot(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
    target: str,
    ax: plt.Axes,
    **kwargs
) -> None:

    sns.boxplot(data=df, x=cat_col, y=num_col, hue=target, ax=ax, **kwargs)
    ax.set_title(f"Boxplot of {num_col} by {cat_col}, grouped by {target}")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(num_col)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()

# def categorical_and_categorical_scatter(
#     df: pd.DataFrame,
#     cat_col1: str,
#     cat_col2: str,
#     target: str,
#     ax=None,
#     **kwargs
# ):



def categorical_and_categorical_box(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    target: str,
    ax: plt.Axes,
    **kwargs
    
) -> None:

    rate_table = pd.crosstab(
        df[cat_col1], df[cat_col2], values=df[target] == "Missed Payment", aggfunc="mean",
    )
    sns.heatmap(rate_table, annot=True, fmt=".0%", ax=ax, **kwargs)
    ax.set_title(f"{"Missed Payment"} rate by {cat_col1} and {cat_col2}")
    ax.set_xlabel(cat_col2)
    ax.set_ylabel(cat_col1)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()


def categorical_and_categorical_count(
    df: pd.DataFrame,
    cat_col1: str,
    cat_col2: str,
    ax: plt.Axes,
    **kwargs
) -> None:

    count_table = pd.crosstab(df[cat_col1], df[cat_col2])
    sns.heatmap(count_table, annot=True, fmt="d", ax=ax, **kwargs)
    ax.set_title(f"Count of {cat_col1} by {cat_col2}")
    ax.set_xlabel(cat_col2)
    ax.set_ylabel(cat_col1)
    ax.tick_params(axis="x", rotation=90)
    plt.tight_layout()


def main() -> None:
    path = DATA_DIR / "loan_status_recoded.csv"
    df = pd.read_csv(path, low_memory=False)

    NUM_VS_NUM_DIR.mkdir(parents=True, exist_ok=True)
    NUM_VS_CAT_SCAT_DIR.mkdir(parents=True, exist_ok=True)
    NUM_VS_CAT_BOX_DIR.mkdir(parents=True, exist_ok=True)
    CAT_VS_CAT_BOX_DIR.mkdir(parents=True, exist_ok=True)

    majority_class, minority_class = _binary_classes(df, "loan_status")
    palette = {majority_class: CATEGORICAL_PALETTE[0], minority_class: CATEGORICAL_PALETTE[1]}

    num_cols = ("installment", "int_rate", "loan_amount")
    cat_cols = ("term", "state", "grade", "type", "purpose")
    # for num_col1, num_col2 in combinations(num_cols, 2):
    #     fig, ax = plt.subplots()
    #     numerical_and_numerical_scatter(
    #         df, num_col1, num_col2, target="loan_status", ax=ax,
    #         palette=palette, alpha=0.8, s=15, edgecolor="none",
    #     )
    #     ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    #     save_path = NUM_VS_NUM_DIR / f"{num_col1}_vs_{num_col2}_scatter.png"
    #     fig.savefig(save_path, dpi=150, bbox_inches="tight")
    #     plt.close(fig)
    #     print(f"Wrote {save_path}")

    # for cat_col in cat_cols:
    #     for num_col in num_cols:
    #         fig, ax = plt.subplots()
    #         categorical_and_numerical_scatter(
    #             df, cat_col, num_col, target="loan_status", ax=ax,
    #             palette=palette, alpha=0.8, size=3, edgecolor="none",
    #         )
    #         ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    #         save_path = NUM_VS_CAT_SCAT_DIR / f"{cat_col}_vs_{num_col}_scatter.png"
    #         fig.savefig(save_path, dpi=150, bbox_inches="tight")
    #         plt.close(fig)
    #         print(f"Wrote {save_path}")

    # for cat_col in cat_cols:
    #     for num_col in num_cols:
    #         fig, ax = plt.subplots()
    #         categorical_and_numerical_boxplot(
    #             df, cat_col, num_col, target="loan_status", ax=ax,
    #             palette=palette,
    #         )
    #         ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    #         save_path = NUM_VS_CAT_BOX_DIR / f"{cat_col}_vs_{num_col}_boxplot.png"
    #         fig.savefig(save_path, dpi=150, bbox_inches="tight")
    #         plt.close(fig)
    #         print(f"Wrote {save_path}")

    for cat_col1, cat_col2 in combinations(cat_cols, 2):
        fig, (ax_rate, ax_count) = plt.subplots(1, 2, figsize=(16, 6))
        categorical_and_categorical_box(df, cat_col1, cat_col2, target="loan_status", ax=ax_rate)
        categorical_and_categorical_count(df, cat_col1, cat_col2, ax=ax_count)
        plt.show()
        save_path = CAT_VS_CAT_BOX_DIR / f"{cat_col1}_vs_{cat_col2}_box.png"
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {save_path}")


if __name__ == "__main__":
    main()

